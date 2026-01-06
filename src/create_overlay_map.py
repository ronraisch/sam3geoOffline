import logging
import base64
import io
from pathlib import Path
from typing import List, Union, Optional, Tuple, NamedTuple
from dataclasses import dataclass

import numpy as np
from PIL import Image
import rasterio
from rasterio.transform import Affine
import leafmap
from ipyleaflet import ImageOverlay, LayersControl
from pyproj import Transformer
from shapely.geometry import Polygon

# --- Configuration ---
DEFAULT_MAX_DIMENSION = 1500
WGS84_CRS = "EPSG:4326"


class MapBounds(NamedTuple):
    """
    NamedTuple for easier debugging and access.
    Format compatible with ipyleaflet: [[south, west], [north, east]]
    """

    south: float
    west: float
    north: float
    east: float

    def to_leaflet(self) -> List[List[float]]:
        """
        Returns the format expected by ImageOverlay.
        SW and NE corners of the image.
        """
        return [
            [float(self.south), float(self.west)],
            [float(self.north), float(self.east)],
        ]


@dataclass
class OverlayConfig:
    path: Union[Path, str]
    name: Optional[str] = None
    opacity: float = 1.0

    @property
    def bounds(self) -> MapBounds:
        try:
            bounds, _ = get_geotiff_footprint(Path(self.path))
            return bounds
        except Exception as e:
            logging.error(f"Error getting bounds for {self.path}: {e}")
            raise e


def get_transformer(src_crs: str) -> Transformer:
    """Creates a transformer from source CRS to WGS84."""
    return Transformer.from_crs(src_crs, WGS84_CRS, always_xy=True)


def load_and_resize_image(path: Path, max_dim: int) -> Tuple[np.ndarray, int, int]:
    """Reads, resizes, and prepares image data for encoding."""
    with Image.open(path) as img:
        if img.mode not in ("RGB", "RGBA"):
            img = img.convert("RGB")

        w, h = img.size
        if max(w, h) > max_dim:
            factor = max_dim / max(w, h)
            img = img.resize(
                (int(w * factor), int(h * factor)), Image.Resampling.LANCZOS
            )

        return np.array(img), img.height, img.width


def get_geotiff_footprint(path: Path) -> Tuple[MapBounds, List[List[float]]]:
    """
    Extracts the geographic footprint.
    Includes a sanity check to prevent Lat/Lon swapping.
    """
    with rasterio.open(path) as src:
        transformer = get_transformer(src.crs)
        h, w = src.height, src.width

        # 1. Get exact corners in world coordinates
        pixel_corners = [(0, 0), (0, w), (h, w), (h, 0)]
        world_pts = [src.xy(r, c) for r, c in pixel_corners]

        # 2. Transform to WGS84 (always_xy=True usually results in Lon, Lat)
        wgs_pts = [transformer.transform(x, y) for x, y in world_pts]

        # 3. Extract all values
        raw_v1 = [p[0] for p in wgs_pts]  # Usually Lon
        raw_v2 = [p[1] for p in wgs_pts]  # Usually Lat

        # SANITY CHECK: In Israel/Middle East, Lat is ~31, Lon is ~34.
        # We ensure 'lats' are the values around 31 and 'lons' are around 34.
        # This logic assumes the area of interest is roughly consistent.
        if np.mean(raw_v1) < np.mean(raw_v2):
            lats, lons = raw_v1, raw_v2
        else:
            lons, lats = raw_v1, raw_v2

        # 4. Construct Leaflet Points [Lat, Lon]
        leaflet_pts = [[float(lat), float(lon)] for lat, lon in zip(lats, lons)]

        bounds = MapBounds(
            south=min(lats), west=min(lons), north=max(lats), east=max(lons)
        )

        return bounds, leaflet_pts


def encode_image_to_base64(data: np.ndarray) -> str:
    """Encodes a numpy array (RGB/RGBA) to a base64 PNG URI."""
    img = Image.fromarray(data)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    b64 = base64.b64encode(buffer.getvalue()).decode()
    return f"data:image/png;base64,{b64}"


def add_geotiff_layer(
    m: leafmap.Map, config: OverlayConfig, max_dim: int
) -> Optional[MapBounds]:
    """Orchestrates adding a single GeoTIFF layer."""
    path = Path(config.path)
    try:
        bounds, _ = get_geotiff_footprint(path)
        print(f"Adding layer: {config.name or path.stem} with bounds {bounds}")
        img_data, _, _ = load_and_resize_image(path, max_dim)
        uri = encode_image_to_base64(img_data)

        layer = ImageOverlay(
            url=uri,
            bounds=bounds.to_leaflet(),
            name=config.name or path.stem,
            opacity=config.opacity,
        )
        m.add_layer(layer)
        return bounds
    except Exception as e:
        logging.error(f"Error loading {path}: {e}")
        return None


def create_geotiff_map(
    base_path: Union[Path, str],
    overlays: List[OverlayConfig],
    max_dim: int = DEFAULT_MAX_DIMENSION,
) -> leafmap.Map:
    """Creates a map and fits it to the base GeoTIFF."""
    base_path = Path(base_path)
    bounds, _ = get_geotiff_footprint(base_path)

    center = [
        float(bounds.south + bounds.north) / 2,
        float(bounds.west + bounds.east) / 2,
    ]

    m = leafmap.Map(center=center, zoom=12)
    m.clear_layers()

    add_geotiff_layer(m, OverlayConfig(path=base_path, name="Base Map"), max_dim)
    for cfg in overlays:
        add_geotiff_layer(m, cfg, max_dim)

    m.add_control(LayersControl())
    m.fit_bounds(bounds.to_leaflet())
    return m

import logging
from pathlib import Path
from typing import Tuple, Union, Optional
import base64
import io
from PIL import Image

# Use the standard import; leafmap will handle the backend
import leafmap
from ipyleaflet import ImageOverlay, LayersControl
from src.utils import MAP_BOUNDS

# Types and Constants
MAX_DIMENSION: int = 2000
path_or_str = Union[Path, str]


def resize_if_too_large(img: Image.Image, max_dimension: int) -> Image.Image:
    """Resize image if it exceeds max_dimension"""
    width, height = img.size
    if width > max_dimension or height > max_dimension:
        ratio = min(max_dimension / width, max_dimension / height)
        new_size = (int(width * ratio), int(height * ratio))
        img = img.resize(new_size, Image.Resampling.LANCZOS)
    return img


def convert_to_base64(img: Image.Image) -> str:
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    img.save(buffered, format="PNG", optimize=True)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def resize_and_encode(
    image_path: path_or_str, max_dimension: int = MAX_DIMENSION
) -> str:
    """Resize image and convert to base64"""
    img = Image.open(image_path)
    img = resize_if_too_large(img, max_dimension)
    return convert_to_base64(img)


def create_offline_leafmap(
    base_map_path: path_or_str,
    base_map_bounds: MAP_BOUNDS,
    overlay_image_path: path_or_str,
    overlay_image_bounds: MAP_BOUNDS,
    max_dimension: int = MAX_DIMENSION,
) -> leafmap.Map:
    # 1. Prepare Image URIs
    base_map_uri = resize_and_encode(base_map_path, max_dimension)
    overlay_uri = resize_and_encode(overlay_image_path, max_dimension)

    # 2. Calculate Center Point
    center_lat = (base_map_bounds[0][0] + base_map_bounds[1][0]) / 2
    center_lon = (base_map_bounds[0][1] + base_map_bounds[1][1]) / 2

    # 3. Initialize Map
    # toolbar_control=True ensures the drawing tools are visible
    m = leafmap.Map(
        center=(center_lat, center_lon),
        zoom=14,
        toolbar_control=True,
        draw_control=True,
        measure_control=True,
        fullscreen_control=True,
    )

    # 4. Clear all default online layers for offline use
    m.clear_layers()

    # 5. Add Static Base Layer
    base_layer = ImageOverlay(
        url=base_map_uri, bounds=base_map_bounds, name="Base Map", opacity=1.0
    )
    m.add_layer(base_layer)

    # 6. Add Static Overlay Layer
    overlay_layer = ImageOverlay(
        url=overlay_uri, bounds=overlay_image_bounds, name="Overlay", opacity=0.6
    )
    m.add_layer(overlay_layer)

    # 7. Add Layer Control
    m.add_control(LayersControl())

    # 8. Focus map
    m.fit_bounds(base_map_bounds)

    return m


if __name__ == "__main__":
    map_bounds = [
        [31.550854962060072, 34.46861743927003],
        [31.572795239267688, 34.51196193695069],
    ]
    overlay_bounds = [
        [31.559083170805593, 34.484871625900276],
        [31.564568240127038, 34.49570775032044],
    ]

    base_map_path = Path("data/map.png")
    overlay_image_path = Path("data/overlay.png")

    m_leafmap = create_offline_leafmap(
        base_map_path=base_map_path,
        base_map_bounds=map_bounds,
        overlay_image_path=overlay_image_path,
        overlay_image_bounds=overlay_bounds,
        max_dimension=500,
    )

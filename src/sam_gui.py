import logging
import io
import base64
import tempfile
import sys
import traceback
from pathlib import Path
from typing import Tuple, Union, Any, Optional, List, Dict, Final

import numpy as np
import rasterio
import shapely
import geopandas as gpd
from PIL import Image
from rasterio import features
import matplotlib.pyplot as plt
from shapely.geometry import shape

import ipyleaflet
import ipywidgets as widgets
import leafmap
import leafmap.colormaps as cm
from samgeo import SamGeo3
from IPython.display import display

# Custom modularized imports
from src.utils import generate_id, write_logs
from buildingregulariser import regularize_geodataframe

from src.create_overlay_map import MapBounds

# --- Configuration & Constants ---


class Config:
    DEFAULT_OPACITY: Final[float] = 0.7
    DEFAULT_BOX_THRESH: Final[float] = 0.25
    DEFAULT_TEXT_THRESH: Final[float] = 0.25
    NODATA_VAL: Final[int] = 0
    ALPHA_FULL: Final[int] = 255

    WIDGET_WIDTH: Final[str] = "100%"
    INPUT_WIDTH: Final[str] = "350px"
    BTN_WIDTH: Final[str] = "120px"
    ICON_SIZE: Final[str] = "28px"
    CURSOR_STYLE: Final[str] = "crosshair"

    TIF_EXT: Final[str] = ".tif"
    GPKG_EXT: Final[str] = ".gpkg"
    RECT_SUFFIX: Final[str] = "_rect"
    PALLETE_DEFAULT: Final[str] = "viridis"


# --- Helper Classes ---


class MapWrapper(leafmap.Map):
    def __init__(self, m: leafmap.Map, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__.update(m.__dict__)
        self.layer_name: str = ""


# --- Core GIS Logic ---


def get_rgba_uri(tif_path: Path, palette: str, opacity: float) -> str:
    with Image.open(tif_path) as img:
        data = np.array(img).astype(float)

    mask = np.where(data == Config.NODATA_VAL, 0, Config.ALPHA_FULL).astype(np.uint8)
    valid = data != Config.NODATA_VAL
    norm = (
        (data - data[valid].min()) / (data[valid].max() - data[valid].min() or 1)
        if valid.any()
        else data
    )

    rgba = (plt.get_cmap(palette)(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = mask

    buf = io.BytesIO()
    Image.fromarray(rgba, "RGBA").save(buf, format="PNG")
    return f"data:image/png;base64,{base64.b64encode(buf.getvalue()).decode()}"


def create_gdf_from_shapes(geoms: List[Dict], crs: Any) -> Optional[gpd.GeoDataFrame]:
    if not geoms:
        return None
    return gpd.GeoDataFrame(geoms, geometry=[g["geometry"] for g in geoms], crs=crs)


def raster_to_vector(src_path: Path, out_path: Path, crs: Any = "EPSG:4326") -> bool:
    with rasterio.open(src_path) as src:
        band = src.read(1)
        src_crs = src.crs or crs
        shape_gen = features.shapes(band, mask=(band != 0), transform=src.transform)
        geoms = [{"geometry": shape(s), "val": v} for s, v in shape_gen]

    gdf = create_gdf_from_shapes(geoms, src_crs)
    if gdf is None:
        return False

    # Filter out invalid or very small geometries before saving
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    try:
        import fiona

        gdf.to_file(out_path)
    except ImportError:
        try:
            gdf.to_file(out_path, engine="pyogrio")
        except:
            pass

    return True


# --- GUI Manager ---


class SAMGuiManager:
    def __init__(self, sam: SamGeo3, m: MapWrapper, bounds: MapBounds, out_dir: Path):
        self.sam, self.m, self.bounds, self.out_dir = sam, m, bounds, out_dir
        self.generated_layers: List[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        self.output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="200px",
                overflow="auto",
                border="1px solid #ccc",
                margin="10px 0",
                padding="5px",
            )
        )
        self.output.add_class("custom-logs")

        self.prompt = widgets.Text(
            description="Prompt:",
            placeholder="e.g. building",
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )
        self.box_slid = self._create_slider("Box Thresh:", Config.DEFAULT_BOX_THRESH)
        self.text_slid = self._create_slider("Text Thresh:", Config.DEFAULT_TEXT_THRESH)
        self.opac_slid = self._create_slider("Opacity:", Config.DEFAULT_OPACITY)
        self.cmap_drop = Config.PALLETE_DEFAULT

        self.reg_check = widgets.Checkbox(
            description="Regularize", value=False, indent=False
        )
        self.color_pick = widgets.ColorPicker(
            description="Color",
            value="#ffff00",
            layout=widgets.Layout(width="180px"),
            style={"description_width": "50px"},
        )

        self.btn_seg = widgets.Button(
            description="Segment",
            button_style="primary",
            layout=widgets.Layout(width=Config.BTN_WIDTH),
        )
        self.btn_reset = widgets.Button(
            description="Reset Layers",
            button_style="warning",
            layout=widgets.Layout(width=Config.BTN_WIDTH),
        )
        self.btn_clear = widgets.Button(
            description="Clear Logs", layout=widgets.Layout(width=Config.BTN_WIDTH)
        )

        self.btn_seg.on_click(self._on_segment_click)
        self.btn_reset.on_click(self._on_reset_click)
        self.btn_clear.on_click(lambda _: self.output.clear_output())

        display(
            widgets.HTML(
                "<style>.custom-logs { white-space: pre-wrap !important; word-wrap: break-word; font-family: monospace; background-color: #f9f9f9; }</style>"
            )
        )

    def _create_slider(self, label: str, val: float) -> widgets.FloatSlider:
        return widgets.FloatSlider(
            description=label,
            min=0,
            max=1,
            step=0.01,
            value=val,
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )

    def _log(self, msg: str, err: bool = False) -> None:
        prefix = "❌" if err else "ℹ️"
        print(f"{prefix} {msg}", file=sys.__stdout__)
        with self.output:
            print(f"{prefix} {msg}")
            write_logs(msg)

    def _get_active_roi(self) -> Optional[List[float]]:
        bounds = self.m.user_roi_bounds()
        if bounds:
            return bounds
        for c in self.m.controls:
            if isinstance(c, ipyleaflet.DrawControl) and c.data:
                return list(shape(c.data[-1].get("geometry")).bounds)
        return None

    def _on_segment_click(self, _):
        roi = self._get_active_roi()
        if not self.prompt.value and not roi:
            return self._log("No prompt or ROI detected.", True)

        try:
            self._execute_segmentation(roi)
        except Exception as e:
            self._log(f"Process error: {str(e)}", True)
            traceback.print_exc(file=sys.__stdout__)

    def _execute_segmentation(self, roi: Optional[List[float]]) -> None:
        prompt_val = self.prompt.value.strip()
        name = f"{prompt_val.replace(' ', '_') or 'mask'}_{generate_id()}"
        tif_path = self.out_dir / f"{name}{Config.TIF_EXT}"

        self.sam.confidence_threshold = self.box_slid.value
        self.sam.mask_threshold = self.text_slid.value

        has_crs = False
        if hasattr(self.sam, "source") and self.sam.source is not None:
            try:
                with rasterio.open(self.sam.source) as src:
                    if src.crs is not None:
                        has_crs = True
            except:
                pass

        if prompt_val:
            self._log(f"Inference via prompt: '{prompt_val}'")
            self.sam.generate_masks(prompt=prompt_val)
        elif roi is not None:
            self._log(f"Inference via ROI box: {roi}")
            b_crs = "EPSG:4326" if has_crs else None
            self.sam.generate_masks_by_boxes(boxes=[roi], box_crs=b_crs)

        self.sam.save_masks(output=str(tif_path))
        if not self._is_mask_valid(tif_path):
            return self._log("SAM found no objects.", True)

        self._add_raster_layer(tif_path, name)
        if self.reg_check.value:
            self._perform_regularization(tif_path, name)

    def _is_mask_valid(self, path: Path) -> bool:
        with rasterio.open(path) as src:
            return bool(np.any(src.read(1) > 0))

    def _add_raster_layer(self, path: Path, name: str) -> None:
        uri = get_rgba_uri(path, self.cmap_drop, self.opac_slid.value)
        layer = ipyleaflet.ImageOverlay(
            url=uri,
            bounds=self.bounds.to_leaflet(),
            name=name,
            opacity=self.opac_slid.value,
        )
        self.m.add_layer(layer)
        self.generated_layers.append(name)
        self.m.layer_name = name
        self._log(f"Success: Layer '{name}' added.")

    def _perform_regularization(self, path: Path, name: str) -> None:
        self._log("Regularizing...")
        v_path = path.with_suffix(Config.GPKG_EXT)

        if not raster_to_vector(path, v_path):
            return self._log("Vectorization failed to produce shapes.", True)

        try:
            gdf = gpd.read_file(v_path)
            if gdf.empty:
                return self._log("No vector features found to regularize.", True)

            # Ensure valid geometries
            gdf = gdf[gdf.is_valid & ~gdf.is_empty]

            # Check for CRS and provide fallback if None
            if gdf.crs is None:
                self._log("Input GDF has no CRS. Assuming EPSG:4326.")
                gdf.set_crs(epsg=4326, inplace=True)

            original_crs = gdf.crs
            assert (
                original_crs is not None
            ), "CRS should not be None for regularization."

            # Regularization often works best in meters (Projected CRS)
            # We'll project to EPSG:3857 (Web Mercator) for processing
            gdf_m = gdf.to_crs(epsg=3857)

            self._log(f"Processing {len(gdf_m)} features...")
            reg_gdf_m = regularize_geodataframe(gdf_m)

            if reg_gdf_m is None or reg_gdf_m.empty:
                # If metric regularization fails, try the original degree-based one as fallback
                self._log("Metric regularization yielded nothing, trying fallback...")
                reg_gdf = regularize_geodataframe(gdf)
            else:
                # Convert back to original CRS for map compatibility

                reg_gdf = reg_gdf_m.to_crs(original_crs)

            if reg_gdf is None or reg_gdf.empty:
                return self._log("Regularization returned empty results.", True)

            self._log(f"Adding regularized layer...")

            try:
                self.m.add_gdf(
                    reg_gdf,
                    layer_name=f"{name}_reg",
                    style={
                        "color": self.color_pick.value,
                        "fillOpacity": 0.5,
                        "weight": 2,
                    },
                )
                self.generated_layers.append(f"{name}_reg")
                self._log("Regularization complete.")
            except Exception as map_err:
                self._log(f"Map rendering error: {str(map_err)}", True)

        except Exception as e:
            self._log(f"Regularization logic error: {str(e)}", True)
            traceback.print_exc(file=sys.__stdout__)

    def _on_reset_click(self, _):
        for n in self.generated_layers:
            layer = self.m.find_layer(n)
            if layer:
                self.m.remove_layer(layer)
        self.generated_layers.clear()
        self.m.layer_name = ""
        self._log("Layers reset.")


# --- UI Entry ---


def text_sam_gui(
    sam: SamGeo3,
    m: MapWrapper,
    overlay_bounds: MapBounds,
    out_dir: Optional[Path] = None,
) -> widgets.VBox:
    """
    Returns a VBox containing the Map at the top and the SAM Controls/Logs below.
    """
    out = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out.mkdir(parents=True, exist_ok=True)
    gui = SAMGuiManager(sam, m, overlay_bounds, out)

    # UI Panel Layout
    controls_box = widgets.VBox(
        [
            widgets.HTML("<h3>SAM v3 Controls</h3>"),
            widgets.HBox(
                [
                    widgets.VBox([gui.prompt, gui.box_slid, gui.text_slid]),
                    widgets.VBox(
                        [
                            gui.opac_slid,
                            widgets.HBox([gui.reg_check, gui.color_pick]),
                        ]
                    ),
                ]
            ),
            widgets.HBox(
                [gui.btn_seg, gui.btn_reset, gui.btn_clear],
                layout=widgets.Layout(margin="10px 0"),
            ),
            gui.output,
        ],
        layout=widgets.Layout(padding="15px", border="1px solid #ddd", margin="10px 0"),
    )

    # Return Map and Controls in a vertical layout
    return widgets.VBox([m, controls_box])

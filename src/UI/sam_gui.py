import logging
import io
import base64
import os
import tempfile
import sys
import traceback
from pathlib import Path
from typing import Tuple, Union, Any, Optional, List, Dict, Final, Iterable

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
from src.UI.create_overlay_map import MapBounds, OverlayConfig, create_geotiff_map

# --- Configuration & Constants ---


class Config:
    # SAM Defaults
    DEFAULT_OPACITY: Final[float] = 0.7
    DEFAULT_BOX_THRESH: Final[float] = 0.25
    DEFAULT_TEXT_THRESH: Final[float] = 0.25

    # Raster Processing
    NODATA_VAL: Final[int] = 0
    ALPHA_FULL: Final[int] = 255
    PNG_FORMAT: Final[str] = "PNG"
    RGBA_MODE: Final[str] = "RGBA"

    # GIS Constants
    WGS84_CRS: Final[str] = "EPSG:4326"
    WEB_MERCATOR_CRS: Final[str] = "EPSG:3857"

    # UI Layout
    WIDGET_WIDTH: Final[str] = "100%"
    INPUT_WIDTH: Final[str] = "350px"
    BTN_WIDTH: Final[str] = "180px"
    COLOR_PICK_WIDTH: Final[str] = "180px"
    LOG_HEIGHT: Final[str] = "200px"

    # File Extensions
    TIF_EXT: Final[str] = ".tif"
    GPKG_EXT: Final[str] = ".gpkg"

    # Visuals
    PALLETE_DEFAULT: Final[str] = "viridis"
    DEFAULT_SEG_COLOR: Final[str] = "#ffff00"
    REG_WEIGHT: Final[int] = 2
    REG_OPACITY: Final[float] = 0.5

    # Layer Filtering
    EXCLUDED_LAYER_NAME: Final[str] = "Base Map"


# --- Core GIS & Image Utilities ---


def convert_tif_to_base64_png(tif_path: Path, palette: str, opacity: float) -> str:
    """Converts a single-band raster to a colormapped RGBA PNG base64 string."""
    with Image.open(tif_path) as img:
        data = np.array(img).astype(float)

    valid_mask = data != Config.NODATA_VAL
    alpha = np.where(valid_mask, Config.ALPHA_FULL, 0).astype(np.uint8)

    if valid_mask.any():
        vmin, vmax = data[valid_mask].min(), data[valid_mask].max()
        norm = (data - vmin) / (vmax - vmin or 1.0)
    else:
        norm = data

    rgba = (plt.get_cmap(palette)(norm) * 255).astype(np.uint8)
    rgba[:, :, 3] = alpha

    buf = io.BytesIO()
    Image.fromarray(rgba, Config.RGBA_MODE).save(buf, format=Config.PNG_FORMAT)
    encoded = base64.b64encode(buf.getvalue()).decode()
    return f"data:image/png;base64,{encoded}"


def vectorise_raster(
    src_path: Path, out_path: Path, crs: str = Config.WGS84_CRS
) -> bool:
    """Converts raster mask to a vectorized GeoPackage file."""
    with rasterio.open(src_path) as src:
        band = src.read(1)
        src_crs = src.crs or crs
        mask = band != Config.NODATA_VAL
        shape_gen = features.shapes(band, mask=mask, transform=src.transform)
        geoms = [{"geometry": shape(s), "val": v} for s, v in shape_gen]

    if not geoms:
        return False

    gdf = gpd.GeoDataFrame(geoms, geometry=[g["geometry"] for g in geoms], crs=src_crs)
    gdf = gdf[gdf.is_valid & ~gdf.is_empty]

    try:
        gdf.to_file(out_path, engine="pyogrio" if "pyogrio" in sys.modules else None)
        return True
    except Exception:
        return False


# --- GUI Manager ---


class SAMGuiManager:
    def __init__(
        self, sam: SamGeo3, m: leafmap.Map, overlays: List[OverlayConfig], out_dir: Path
    ):
        self.sam = sam
        self.m = m
        self.overlays = overlays
        self.out_dir = out_dir
        self.generated_layers: List[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        """Initializes and displays the widget dashboard."""
        self.output = widgets.Output(
            layout=widgets.Layout(
                width=Config.WIDGET_WIDTH,
                height=Config.LOG_HEIGHT,
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
        self.box_slid = widgets.FloatSlider(
            description="Box Thresh:",
            min=0,
            max=1,
            step=0.01,
            value=Config.DEFAULT_BOX_THRESH,
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )
        self.text_slid = widgets.FloatSlider(
            description="Text Thresh:",
            min=0,
            max=1,
            step=0.01,
            value=Config.DEFAULT_TEXT_THRESH,
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )
        self.opac_slid = widgets.FloatSlider(
            description="Opacity:",
            min=0,
            max=1,
            step=0.01,
            value=Config.DEFAULT_OPACITY,
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )

        self.reg_check = widgets.Checkbox(
            description="Regularize", value=False, indent=False
        )
        self.color_pick = widgets.ColorPicker(
            description="Color",
            value=Config.DEFAULT_SEG_COLOR,
            layout=widgets.Layout(width=Config.COLOR_PICK_WIDTH),
            style={"description_width": "50px"},
        )

        self.btn_seg = widgets.Button(
            description="Segment All Layers",
            button_style="primary",
            layout=widgets.Layout(width="200px"),
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
                "<style>.custom-logs { white-space: pre-wrap; font-family: monospace; background: #f9f9f9; }</style>"
            )
        )

    def _log(self, msg: str, is_error: bool = False) -> None:
        icon = "❌" if is_error else "ℹ️"
        formatted = f"{icon} {msg}"
        print(formatted, file=sys.__stdout__)
        with self.output:
            print(formatted)
            write_logs(msg)

    def _get_active_roi(self) -> Optional[List[float]]:
        """Retrieves user-drawn ROI or active bounds from the map."""
        bounds = self.m.user_roi_bounds()
        if bounds:
            return bounds
        for control in self.m.controls:
            if isinstance(control, ipyleaflet.DrawControl) and control.data:
                return list(shape(control.data[-1].get("geometry")).bounds)
        return None

    def _get_target_layers(self) -> List[ipyleaflet.ImageOverlay]:
        """Identifies all ImageOverlay layers that are not the Base Map."""
        targets = []
        for layer in self.m.layers:
            if isinstance(layer, ipyleaflet.ImageOverlay):
                if layer.name != Config.EXCLUDED_LAYER_NAME:
                    targets.append(layer)
        return targets

    def _on_segment_click(self, _: widgets.Button) -> None:
        prompt_text = self.prompt.value.strip()
        roi = self._get_active_roi()

        if not prompt_text and not roi:
            return self._log(
                "Please enter a text prompt or draw an ROI on the map.", is_error=True
            )

        target_layers = self._get_target_layers()
        if not target_layers:
            return self._log("No target image layers found on the map.", is_error=True)

        mode = "Prompt" if prompt_text else "ROI Box"
        self._log(
            f"Starting batch segmentation via {mode} for {len(target_layers)} layers..."
        )

        for layer in target_layers:
            try:
                overlay_cfg = next(
                    (o for o in self.overlays if o.name == layer.name), None
                )
                if not overlay_cfg:
                    continue

                self._log(f"Processing layer: {layer.name}...")
                self._run_inference_on_layer(overlay_cfg, prompt_text, roi)
            except Exception as e:
                self._log(f"Failed processing '{layer.name}': {str(e)}", is_error=True)

    def _run_inference_on_layer(
        self, overlay: OverlayConfig, prompt: str, roi: Optional[List[float]]
    ) -> None:
        """Runs SAM inference on a specific image source using prompt or ROI."""
        self.sam.set_image(str(overlay.path))

        unique_id = generate_id()
        p_slug = prompt.replace(" ", "_") if prompt else "roi"
        base_name = f"{p_slug}_{overlay.name}_{unique_id}"
        tif_path = self.out_dir / f"{base_name}{Config.TIF_EXT}"

        # Setup SAM params
        self.sam.confidence_threshold = self.box_slid.value
        self.sam.mask_threshold = self.text_slid.value

        # Execute
        if prompt:
            self.sam.generate_masks(prompt=prompt)
        elif roi:
            self.sam.generate_masks_by_boxes(boxes=[roi], box_crs=Config.WGS84_CRS)

        self.sam.save_masks(output=str(tif_path))

        if not self._validate_mask(tif_path):
            self._log(f"No objects detected in {overlay.name}.")
            return

        self._add_raster_to_map(tif_path, base_name, overlay.bounds)

        if self.reg_check.value:
            self._process_regularization(tif_path, base_name)

    def _validate_mask(self, path: Path) -> bool:
        with rasterio.open(path) as src:
            return bool(np.any(src.read(1) > 0))

    def _add_raster_to_map(self, path: Path, name: str, bounds: MapBounds) -> None:
        uri = convert_tif_to_base64_png(
            path, Config.PALLETE_DEFAULT, self.opac_slid.value
        )
        overlay = ipyleaflet.ImageOverlay(
            url=uri,
            bounds=bounds.to_leaflet(),
            name=name,
            opacity=self.opac_slid.value,
        )
        self.m.add_layer(overlay)
        self.generated_layers.append(name)
        self._log(f"Layer '{name}' added.")

    def _process_regularization(self, path: Path, name: str) -> None:
        self._log(f"Regularizing {name}...")
        vec_path = path.with_suffix(Config.GPKG_EXT)

        if not vectorise_raster(path, vec_path):
            return

        try:
            gdf = gpd.read_file(vec_path)
            if gdf.empty:
                return
            if gdf.crs is None:
                gdf.set_crs(epsg=4326, inplace=True)

            original_crs = gdf.crs
            assert original_crs is not None, "CRS should not be None."
            gdf_m = gdf.to_crs(epsg=3857)
            reg_gdf_m = regularize_geodataframe(gdf_m)

            if reg_gdf_m is not None and not reg_gdf_m.empty:
                reg_gdf = reg_gdf_m.to_crs(original_crs)
                self._add_regularized_to_map(reg_gdf, name)
        except Exception as e:
            self._log(f"Regularization Error: {str(e)}", is_error=True)

    def _add_regularized_to_map(self, gdf: gpd.GeoDataFrame, base_name: str) -> None:
        layer_name = f"{base_name}_reg"
        self.m.add_gdf(
            gdf,
            layer_name=layer_name,
            style={
                "color": self.color_pick.value,
                "fillOpacity": Config.REG_OPACITY,
                "weight": Config.REG_WEIGHT,
            },
        )
        self.generated_layers.append(layer_name)

    def _on_reset_click(self, _: widgets.Button) -> None:
        for name in self.generated_layers:
            layer = self.m.find_layer(name)
            if layer:
                self.m.remove_layer(layer)
        self.generated_layers.clear()
        self._log("Results cleared.")


def text_sam_gui(
    sam: SamGeo3,
    overlays: List[OverlayConfig],
    base_map_path: Union[Path, str],
    out_dir: Optional[Path] = None,
) -> widgets.VBox:
    temp_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)

    m = create_geotiff_map(base_map_path, overlays)
    gui = SAMGuiManager(sam, m, overlays, temp_dir)

    panel = widgets.VBox(
        [
            widgets.HTML("<h3>SAM v3 Batch Processing</h3>"),
            widgets.HTML(
                "<p style='font-size:0.9em; color:#666'>Enter a prompt OR draw a box on the map to segment all layers.</p>"
            ),
            widgets.HBox(
                [
                    widgets.VBox([gui.prompt, gui.box_slid, gui.text_slid]),
                    widgets.VBox(
                        [gui.opac_slid, widgets.HBox([gui.reg_check, gui.color_pick])]
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

    return widgets.VBox([m, panel])

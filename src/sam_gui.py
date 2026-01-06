import logging
import io
import base64
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
from src.create_overlay_map import MapBounds

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
    BTN_WIDTH: Final[str] = "120px"
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


# --- Core GIS & Image Utilities ---


def convert_tif_to_base64_png(tif_path: Path, palette: str, opacity: float) -> str:
    """Converts a single-band raster to a colormapped RGBA PNG base64 string."""
    with Image.open(tif_path) as img:
        data = np.array(img).astype(float)

    valid_mask = data != Config.NODATA_VAL
    alpha = np.where(valid_mask, Config.ALPHA_FULL, 0).astype(np.uint8)

    # Normalize data for colormap
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
        # Prioritize engines that don't require system-level fiona/gdal setups
        gdf.to_file(out_path, engine="pyogrio" if "pyogrio" in sys.modules else None)
        return True
    except Exception:
        return False


# --- UI Component Factory ---


class UIFactory:
    @staticmethod
    def create_slider(label: str, val: float) -> widgets.FloatSlider:
        return widgets.FloatSlider(
            description=label,
            min=0,
            max=1,
            step=0.01,
            value=val,
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )

    @staticmethod
    def create_button(desc: str, style: str = "") -> widgets.Button:
        return widgets.Button(
            description=desc,
            button_style=style,
            layout=widgets.Layout(width=Config.BTN_WIDTH),
        )


# --- Map & Layer Wrapper ---


class MapWrapper(leafmap.Map):
    def __init__(self, m: leafmap.Map, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__.update(m.__dict__)


# --- GUI Manager ---


class SAMGuiManager:
    def __init__(self, sam: SamGeo3, m: MapWrapper, bounds: MapBounds, out_dir: Path):
        self.sam = sam
        self.m = m
        self.bounds = bounds
        self.out_dir = out_dir
        self.generated_layers: List[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        """Initializes and displays the widget dashboard."""
        # Logs Output
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

        # Inputs
        self.prompt = widgets.Text(
            description="Prompt:",
            placeholder="e.g. building",
            layout=widgets.Layout(width=Config.INPUT_WIDTH),
        )
        self.box_slid = UIFactory.create_slider(
            "Box Thresh:", Config.DEFAULT_BOX_THRESH
        )
        self.text_slid = UIFactory.create_slider(
            "Text Thresh:", Config.DEFAULT_TEXT_THRESH
        )
        self.opac_slid = UIFactory.create_slider("Opacity:", Config.DEFAULT_OPACITY)

        self.reg_check = widgets.Checkbox(
            description="Regularize", value=False, indent=False
        )
        self.color_pick = widgets.ColorPicker(
            description="Color",
            value=Config.DEFAULT_SEG_COLOR,
            layout=widgets.Layout(width=Config.COLOR_PICK_WIDTH),
            style={"description_width": "50px"},
        )

        # Buttons
        self.btn_seg = UIFactory.create_button("Segment", "primary")
        self.btn_reset = UIFactory.create_button("Reset Layers", "warning")
        self.btn_clear = UIFactory.create_button("Clear Logs")

        self.btn_seg.on_click(self._on_segment_click)
        self.btn_reset.on_click(self._on_reset_click)
        self.btn_clear.on_click(lambda _: self.output.clear_output())

        # Global CSS for output
        display(
            widgets.HTML(
                "<style>.custom-logs { white-space: pre-wrap; font-family: monospace; background: #f9f9f9; }</style>"
            )
        )

    def _log(self, msg: str, is_error: bool = False) -> None:
        """Unified logging to UI, terminal, and file."""
        icon = "❌" if is_error else "ℹ️"
        formatted = f"{icon} {msg}"
        print(formatted, file=sys.__stdout__)
        with self.output:
            print(formatted)
            write_logs(msg)

    def _get_active_roi(self) -> Optional[List[float]]:
        """Retrieves user-drawn ROI or active bounds."""
        bounds = self.m.user_roi_bounds()
        if bounds:
            return bounds
        for control in self.m.controls:
            if isinstance(control, ipyleaflet.DrawControl) and control.data:
                return list(shape(control.data[-1].get("geometry")).bounds)
        return None

    def _on_segment_click(self, _: widgets.Button) -> None:
        roi = self._get_active_roi()
        if not self.prompt.value.strip() and not roi:
            return self._log("Missing prompt or ROI selection.", is_error=True)

        try:
            self._run_inference_pipeline(roi)
        except Exception as e:
            self._log(f"Pipeline failed: {str(e)}", is_error=True)
            traceback.print_exc(file=sys.__stdout__)

    def _run_inference_pipeline(self, roi: Optional[List[float]]) -> None:
        """Handles mask generation, saving, and rendering."""
        prompt_text = self.prompt.value.strip()
        unique_id = generate_id()
        base_name = f"{prompt_text.replace(' ', '_') or 'mask'}_{unique_id}"
        tif_path = self.out_dir / f"{base_name}{Config.TIF_EXT}"

        # Setup SAM params
        self.sam.confidence_threshold = self.box_slid.value
        self.sam.mask_threshold = self.text_slid.value

        # Execute segmentation
        if prompt_text:
            self._log(f"Segmenting prompt: '{prompt_text}'")
            self.sam.generate_masks(prompt=prompt_text)
        elif roi:
            self._log(f"Segmenting ROI box: {roi}")
            self.sam.generate_masks_by_boxes(boxes=[roi], box_crs=Config.WGS84_CRS)

        self.sam.save_masks(output=str(tif_path))

        if not self._validate_mask(tif_path):
            return self._log(
                "No objects detected in the specified area.", is_error=True
            )

        self._add_raster_to_map(tif_path, base_name)

        if self.reg_check.value:
            self._process_regularization(tif_path, base_name)

    def _validate_mask(self, path: Path) -> bool:
        """Returns True if the raster contains non-zero data."""
        with rasterio.open(path) as src:
            return bool(np.any(src.read(1) > 0))

    def _add_raster_to_map(self, path: Path, name: str) -> None:
        """Generates visual overlay and adds to leaflet map."""
        uri = convert_tif_to_base64_png(
            path, Config.PALLETE_DEFAULT, self.opac_slid.value
        )
        overlay = ipyleaflet.ImageOverlay(
            url=uri,
            bounds=self.bounds.to_leaflet(),
            name=name,
            opacity=self.opac_slid.value,
        )
        self.m.add_layer(overlay)
        self.generated_layers.append(name)
        self._log(f"Layer '{name}' added.")

    def _process_regularization(self, path: Path, name: str) -> None:
        """Converts raster to vector and applies building regularization."""
        self._log("Initiating regularization...")
        vec_path = path.with_suffix(Config.GPKG_EXT)

        if not vectorise_raster(path, vec_path):
            return self._log(
                "Vectorization yielded no valid geometries.", is_error=True
            )

        try:
            gdf = gpd.read_file(vec_path)
            if gdf.empty:
                return

            reg_gdf = self._apply_regularization_logic(gdf)
            if reg_gdf is None or reg_gdf.empty:
                return self._log("Regularization returned no results.", is_error=True)

            self._add_regularized_to_map(reg_gdf, name)

        except Exception as e:
            self._log(f"Regularization Error: {str(e)}", is_error=True)

    def _apply_regularization_logic(
        self, gdf: gpd.GeoDataFrame
    ) -> Optional[gpd.GeoDataFrame]:
        """Handles CRS projection and regularization algorithm calls."""
        if gdf.crs is None:
            gdf.set_crs(epsg=4326, inplace=True)

        original_crs = gdf.crs

        # Regularize in Metric Space (Projected CRS)
        gdf_m = gdf.to_crs(epsg=3857)
        self._log(f"Regularizing {len(gdf_m)} features...")
        reg_gdf_m = regularize_geodataframe(gdf_m)

        if reg_gdf_m is None or reg_gdf_m.empty:
            self._log("Metric regularization failed, trying standard fallback...")
            return regularize_geodataframe(gdf)
        assert original_crs is not None, "Original CRS should not be None"
        return reg_gdf_m.to_crs(original_crs)

    def _add_regularized_to_map(self, gdf: gpd.GeoDataFrame, base_name: str) -> None:
        """Adds the vectorized regularized GeoDataFrame to the leaflet map."""
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
        self._log("Regularization complete.")

    def _on_reset_click(self, _: widgets.Button) -> None:
        """Removes all generated layers from the map."""
        for name in self.generated_layers:
            layer = self.m.find_layer(name)
            if layer:
                self.m.remove_layer(layer)
        self.generated_layers.clear()
        self._log("Workspace cleared.")


# --- Entry Point ---


def text_sam_gui(
    sam: SamGeo3,
    m: MapWrapper,
    overlay_bounds: MapBounds,
    out_dir: Optional[Path] = None,
) -> widgets.VBox:
    """Entry point to create the SAM v3 Integrated UI."""
    temp_dir = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    temp_dir.mkdir(parents=True, exist_ok=True)

    gui = SAMGuiManager(sam, m, overlay_bounds, temp_dir)

    panel = widgets.VBox(
        [
            widgets.HTML("<h3>SAM v3 Controls</h3>"),
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

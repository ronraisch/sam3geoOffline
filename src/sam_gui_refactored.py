import logging
import io
import base64
import random
import string
import subprocess
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

# --- Configuration & Constants ---


class Config:
    DEFAULT_OPACITY: Final[float] = 0.7
    DEFAULT_BOX_THRESH: Final[float] = 0.25
    DEFAULT_TEXT_THRESH: Final[float] = 0.25
    NODATA_VAL: Final[int] = 0
    ALPHA_FULL: Final[int] = 255

    WIDGET_WIDTH: Final[str] = "280px"
    BTN_WIDTH: Final[str] = "90px"
    ICON_SIZE: Final[str] = "28px"
    CURSOR_STYLE: Final[str] = "crosshair"

    TIF_EXT: Final[str] = ".tif"
    GPKG_EXT: Final[str] = ".gpkg"
    RECT_SUFFIX: Final[str] = "_rect"


MAP_BOUNDS = Union[List[List[float]], Tuple[Tuple[float, float], Tuple[float, float]]]

# --- Helper Classes ---


class MapWrapper(leafmap.Map):
    def __init__(self, m: leafmap.Map, **kwargs: Any):
        super().__init__(**kwargs)
        self.__dict__.update(m.__dict__)
        self.layer_name: str = ""


# --- System Utilities ---


def generate_id(length: int = 6) -> str:
    return "".join(random.choice(string.ascii_lowercase) for _ in range(length))


def install_package(package: str) -> None:
    cmd = (
        f"pip install git+{package}"
        if package.startswith("https://")
        else f"pip install {package}"
    )
    try:
        subprocess.check_call(cmd.split())
    except subprocess.CalledProcessError as e:
        print(f"Installation failed: {e}")


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
    """Safely creates a GeoDataFrame. Returns None if geoms is empty."""
    if not geoms:
        return None

    # We initialize the GDF with the geometry list and CRS simultaneously.
    # This prevents the 'no geometry column' error during CRS assignment.
    return gpd.GeoDataFrame(geoms, geometry=[g["geometry"] for g in geoms], crs=crs)


def raster_to_vector(src_path: Path, out_path: Path, crs: Any = "EPSG:4326") -> bool:
    with rasterio.open(src_path) as src:
        band = src.read(1)
        src_crs = src.crs or crs
        # Extract shapes from the raster band
        shape_gen = features.shapes(band, mask=(band != 0), transform=src.transform)
        geoms = [{"geometry": shape(s), "val": v} for s, v in shape_gen]

    gdf = create_gdf_from_shapes(geoms, src_crs)
    if gdf is None:
        return False

    gdf.to_file(out_path)
    return True


# --- GUI Manager ---


class SAMGuiManager:
    def __init__(self, sam: SamGeo3, m: MapWrapper, bounds: MAP_BOUNDS, out_dir: Path):
        self.sam, self.m, self.bounds, self.out_dir = sam, m, bounds, out_dir
        self.generated_layers: List[str] = []
        self._init_ui()

    def _init_ui(self) -> None:
        self.output = widgets.Output(
            layout=widgets.Layout(
                width="100%",
                height="160px",
                overflow="auto",
                border="1px solid #4A90E2",
                margin="5px 0",
            )
        )
        self.output.add_class("custom-logs")

        self.prompt = widgets.Text(
            description="Prompt:",
            placeholder="e.g. building",
            layout=widgets.Layout(width=Config.WIDGET_WIDTH),
        )
        self.box_slid = self._create_slider("Box Thresh:", Config.DEFAULT_BOX_THRESH)
        self.text_slid = self._create_slider("Text Thresh:", Config.DEFAULT_TEXT_THRESH)
        self.opac_slid = self._create_slider("Opacity:", Config.DEFAULT_OPACITY)
        self.cmap_drop = widgets.Dropdown(
            description="Palette:",
            options=cm.list_colormaps(),
            value="viridis",
            layout=widgets.Layout(width=Config.WIDGET_WIDTH),
        )
        self.reg_check = widgets.Checkbox(
            description="Regularize", value=False, indent=False
        )
        self.color_pick = widgets.ColorPicker(
            description="Color",
            value="#ffff00",
            layout=widgets.Layout(width="130px"),
            style={"description_width": "40px"},
        )

        self.btn_seg = widgets.ToggleButton(
            description="Segment",
            button_style="primary",
            layout=widgets.Layout(width=Config.BTN_WIDTH),
        )
        self.btn_reset = widgets.ToggleButton(
            description="Reset",
            button_style="primary",
            layout=widgets.Layout(width=Config.BTN_WIDTH),
        )
        self.btn_clear = widgets.Button(
            description="Clear Logs",
            layout=widgets.Layout(width=Config.WIDGET_WIDTH, height="24px"),
        )

        self.btn_seg.observe(self._on_segment, "value")
        self.btn_reset.observe(self._on_reset, "value")
        self.btn_clear.on_click(lambda _: self.output.clear_output())

        # CSS to ensure logs wrap correctly and are readable
        display(
            widgets.HTML(
                "<style>.custom-logs { white-space: pre-wrap !important; word-wrap: break-word; font-family: monospace; }</style>"
            )
        )

    def _create_slider(self, label: str, val: float) -> widgets.FloatSlider:
        return widgets.FloatSlider(
            description=label,
            min=0,
            max=1,
            step=0.01,
            value=val,
            layout=widgets.Layout(width=Config.WIDGET_WIDTH),
        )

    def _log(self, msg: str, err: bool = False) -> None:
        prefix = "❌" if err else "ℹ️"
        print(f"{prefix} {msg}", file=sys.__stdout__)
        with self.output:
            print(f"{prefix} {msg}")

    def _get_active_roi(self) -> Optional[List[float]]:
        # Returns [xmin, ymin, xmax, ymax]
        bounds = self.m.user_roi_bounds()
        if bounds:
            return bounds
        for c in self.m.controls:
            if isinstance(c, ipyleaflet.DrawControl) and c.data:
                return list(shape(c.data[-1].get("geometry")).bounds)
        return None

    def _on_segment(self, change: Dict) -> None:
        if not change["new"]:
            return
        self.btn_seg.value = False
        roi = self._get_active_roi()

        if not self.prompt.value and not roi:
            return self._log("No prompt or ROI detected.", True)

        try:
            self._execute_segmentation(roi)
        except Exception as e:
            self._log(f"Process error: {str(e)}", True)
            traceback.print_exc(file=sys.__stdout__)

    def _execute_segmentation(self, roi: Optional[List[float]]) -> None:
        """From-scratch implementation of the segmentation workflow."""
        # 1. Setup Identifiers
        prompt_val = self.prompt.value.strip()
        name = f"{prompt_val.replace(' ', '_') or 'mask'}_{generate_id()}"
        tif_path = self.out_dir / f"{name}{Config.TIF_EXT}"

        # 2. Update Model Config
        self.sam.confidence_threshold = self.box_slid.value
        self.sam.mask_threshold = self.text_slid.value

        # 3. Perform Inference
        if prompt_val:
            self._log(f"Inference via prompt: '{prompt_val}'")
            self.sam.generate_masks(prompt=prompt_val)
        elif roi is not None:
            self._log(f"Inference via ROI box: {roi}")
            # Ensure ROI is passed correctly as a list of boxes
            self.sam.generate_masks_by_boxes(boxes=[roi], box_crs="EPSG:4326")
        else:
            return self._log(
                "Unable to proceed: No prompt and no valid ROI found.", True
            )

        # 4. Save to Disk
        self.sam.save_masks(output=str(tif_path))

        # 5. Result Validation
        if not self._is_mask_valid(tif_path):
            return self._log("SAM found no objects matching the criteria.", True)

        # 6. UI Updates
        self._add_raster_layer(tif_path, name)

        # 7. Post-Processing
        if self.reg_check.value:
            self._perform_regularization(tif_path, name)

    def _is_mask_valid(self, path: Path) -> bool:
        with rasterio.open(path) as src:
            return bool(np.any(src.read(1) > 0))

    def _add_raster_layer(self, path: Path, name: str) -> None:
        uri = get_rgba_uri(path, self.cmap_drop.value, self.opac_slid.value)
        layer = ipyleaflet.ImageOverlay(
            url=uri, bounds=self.bounds, name=name, opacity=self.opac_slid.value
        )
        self.m.add_layer(layer)
        self.generated_layers.append(name)
        self.m.layer_name = name
        self._log(f"Success: Layer '{name}' added.")

    def _perform_regularization(self, path: Path, name: str) -> None:
        self._log("Converting to vector and regularizing...")
        v_path = path.with_suffix(Config.GPKG_EXT)
        v_rect = path.with_name(f"{path.stem}_reg{Config.GPKG_EXT}")

        if not raster_to_vector(path, v_path):
            return self._log("Vectorization yielded no valid shapes.", True)

        try:
            from buildingregulariser import regularize_geodataframe

            gdf = gpd.read_file(v_path)
            reg_gdf = regularize_geodataframe(gdf)
            reg_gdf.to_file(v_rect)

            self.m.add_vector(
                str(v_rect),
                layer_name=f"{name}_reg",
                style={"color": self.color_pick.value},
            )
            self.generated_layers.append(f"{name}_reg")
            self._log("Regularization complete.")
        except ImportError:
            self._log("Module 'buildingregulariser' not found.", True)

    def _on_reset(self, change: Dict) -> None:
        if not change["new"]:
            return
        self.btn_reset.value = False
        for n in self.generated_layers:
            layer = self.m.find_layer(n)
            if layer:
                self.m.remove_layer(layer)
        self.generated_layers.clear()
        self.output.clear_output()
        self.m.layer_name = ""
        self._log("Reset complete.")


# --- UI Entry ---


def text_sam_gui(
    sam: SamGeo3,
    m: MapWrapper,
    overlay_bounds: MAP_BOUNDS,
    out_dir: Optional[Path] = None,
) -> MapWrapper:
    out = Path(out_dir) if out_dir else Path(tempfile.gettempdir())
    out.mkdir(parents=True, exist_ok=True)
    gui = SAMGuiManager(sam, m, overlay_bounds, out)

    ui = widgets.VBox(
        [
            widgets.HBox(
                [
                    widgets.ToggleButton(
                        icon="times",
                        button_style="primary",
                        layout=widgets.Layout(width=Config.ICON_SIZE),
                    ),
                    widgets.ToggleButton(
                        icon="gear",
                        value=True,
                        layout=widgets.Layout(width=Config.ICON_SIZE),
                    ),
                ]
            ),
            widgets.VBox(
                [
                    gui.prompt,
                    gui.box_slid,
                    gui.text_slid,
                    gui.cmap_drop,
                    gui.opac_slid,
                    widgets.HBox(
                        [gui.reg_check, gui.color_pick],
                        layout=widgets.Layout(
                            width=Config.WIDGET_WIDTH, justify_content="space-between"
                        ),
                    ),
                    widgets.HBox([gui.btn_seg, gui.btn_reset]),
                    gui.btn_clear,
                    gui.output,
                ],
                layout=widgets.Layout(padding="8px"),
            ),
        ]
    )

    m.add_control(ipyleaflet.WidgetControl(widget=ui, position="topright"))
    return m

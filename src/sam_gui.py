import logging
import os
from typing import Union
import geopandas as gpd
from typing import Any, Optional
from PIL import Image
import io
import base64
import ipyleaflet
import leafmap
from matplotlib import pyplot as plt
import numpy as np
import pyproj
import rasterio
from samgeo import SamGeo3
from rasterio import features
import shapely
from ipyleaflet import ImageOverlay

from .create_overlay_map import MAP_BOUNDS


class MapWrapper(leafmap.Map):
    """A wrapper around MapWrapper to add custom functionality if needed."""

    def __init__(
        self,
        m: leafmap.Map,
        layer_name: Optional[str] = None,
        toolbar: Optional[ipyleaflet.WidgetControl] = None,
        save_control: Optional[ipyleaflet.WidgetControl] = None,
        **kwargs,
    ):
        super().__init__(**kwargs)
        self.__dict__.update(m.__dict__)
        if layer_name is not None:
            self.layer_name = layer_name
        else:
            self.layer_name = ""
        self.toolbar_control = toolbar
        self.save_control = save_control


def add_offline_raster(
    m: MapWrapper,
    tif_path: str,
    bounds: MAP_BOUNDS,
    palette: str = "viridis",
    nodata: int = 0,
    opacity: float = 1.0,
):
    """
    Applies a palette and nodata transparency to a TIF,
    then adds it to the map as an offline-safe ImageOverlay.
    """
    # 1. Open image with PIL
    with Image.open(tif_path) as img:
        # Convert to numpy array for processing
        data = np.array(img).astype(float)

    # 2. Handle Nodata: Create a mask where data is NOT the nodata value
    # This will be used for the Alpha channel
    alpha_mask = np.where(data == nodata, 0, 255).astype(np.uint8)

    # 3. Apply Palette (Color Mapping)
    # Normalize data to 0-1 for matplotlib colormaps
    valid_mask = data != nodata
    if valid_mask.any():
        data_min, data_max = data[valid_mask].min(), data[valid_mask].max()
        # Avoid division by zero
        if data_max != data_min:
            normalized_data = (data - data_min) / (data_max - data_min)
        else:
            normalized_data = data * 0
    else:
        normalized_data = data

    # Get the colormap from matplotlib
    cmap = plt.get_cmap(palette)
    # Map the normalized data to RGBA (values 0.0 to 1.0)
    colored_data = cmap(normalized_data)

    # Convert to 0-255 scale uint8
    rgba_image = (colored_data * 255).astype(np.uint8)

    # 4. Inject our Nodata Mask into the Alpha channel (index 3)
    rgba_image[:, :, 3] = alpha_mask

    # 5. Convert to Base64 URI
    pill_img = Image.fromarray(rgba_image, "RGBA")

    # Optional: Resize if it exceeds MAX_DIMENSION to keep the map snappy
    # pill_img = resize_if_too_large(pill_img, 2000)

    buffered = io.BytesIO()
    pill_img.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    uri = f"data:image/png;base64,{img_str}"

    # 6. Add to Map
    new_layer = ImageOverlay(url=uri, bounds=bounds, name=m.layer_name, opacity=opacity)

    m.add_layer(new_layer)


def raster_to_vector(source, output, simplify_tolerance=None, dst_crs=None, **kwargs):
    """Vectorize a raster dataset.

    Args:
        source (str): The path to the tiff file.
        output (str): The path to the vector file.
        simplify_tolerance (float, optional): The maximum allowed geometry displacement.
            The higher this value, the smaller the number of vertices in the resulting geometry.
    """

    with rasterio.open(source) as src:
        band = src.read()

        mask = band != 0
        shapes = features.shapes(band, mask=mask, transform=src.transform)

    fc = [
        {"geometry": shapely.geometry.shape(shape), "properties": {"value": value}}
        for shape, value in shapes
    ]
    if simplify_tolerance is not None:
        for i in fc:
            i["geometry"] = i["geometry"].simplify(tolerance=simplify_tolerance)

    gdf = gpd.GeoDataFrame.from_features(fc)
    if src.crs is not None:
        gdf.set_crs(crs=src.crs, inplace=True)

    if dst_crs is not None:
        gdf = gdf.to_crs(dst_crs)

    gdf.to_file(output, **kwargs)


def random_string(string_length=6):
    """Generates a random string of fixed length.

    Args:
        string_length (int, optional): Fixed length. Defaults to 3.

    Returns:
        str: A random string
    """
    import random
    import string

    # random.seed(1001)
    letters = string.ascii_lowercase
    return "".join(random.choice(letters) for i in range(string_length))


def install_package(package):
    """Install a Python package.

    Args:
        package (str | list): The package name or a GitHub URL or a list of package names or GitHub URLs.
    """
    import subprocess

    if isinstance(package, str):
        packages = [package]

    for package in packages:
        if package.startswith("https://github.com"):
            package = f"git+{package}"

        # Execute pip install command and show output in real-time
        command = f"pip install {package}"
        process = subprocess.Popen(command.split(), stdout=subprocess.PIPE)

        # Print output in real-time
        while True:
            if process.stdout is None:
                break
            output = process.stdout.readline()
            if output == b"" and process.poll() is not None:
                break
            if output:
                print(output.decode("utf-8").strip())

        # Wait for process to complete
        process.wait()


def regularize(
    data: Union[gpd.GeoDataFrame, str],
    output_path: Optional[str] = None,
    parallel_threshold: float = 1.0,
    target_crs: Optional[Union[str, "pyproj.CRS"]] = None,
    simplify: bool = True,
    simplify_tolerance: float = 0.5,
    allow_45_degree: bool = True,
    diagonal_threshold_reduction: float = 15,
    allow_circles: bool = True,
    circle_threshold: float = 0.9,
    num_cores: int = 1,
    include_metadata: bool = False,
    **kwargs: Any,
) -> Any:
    """Regularizes polygon geometries in a GeoDataFrame by aligning edges.

    Aligns edges to be parallel or perpendicular (optionally also 45 degrees)
    to their main direction. Handles reprojection, initial simplification,
    regularization, geometry cleanup, and parallel processing.

    This function is a wrapper around the `regularize_geodataframe` function
    from the `buildingregulariser` package. Credits to the original author
    Nick Wright. Check out the repo at https://github.com/DPIRD-DMA/Building-Regulariser.

    Args:
        data (Union[gpd.GeoDataFrame, str]): Input GeoDataFrame with polygon or multipolygon geometries,
            or a file path to the GeoDataFrame.
        output_path (Optional[str], optional): Path to save the output GeoDataFrame. If None, the output is
            not saved. Defaults to None.
        parallel_threshold (float, optional): Distance threshold for merging nearly parallel adjacent edges
            during regularization. Defaults to 1.0.
        target_crs (Optional[Union[str, "pyproj.CRS"]], optional): Target Coordinate Reference System for
            processing. If None, uses the input GeoDataFrame's CRS. Processing is more reliable in a
            projected CRS. Defaults to None.
        simplify (bool, optional): If True, applies initial simplification to the geometry before
            regularization. Defaults to True.
        simplify_tolerance (float, optional): Tolerance for the initial simplification step (if `simplify`
            is True). Also used for geometry cleanup steps. Defaults to 0.5.
        allow_45_degree (bool, optional): If True, allows edges to be oriented at 45-degree angles relative
            to the main direction during regularization. Defaults to True.
        diagonal_threshold_reduction (float, optional): Reduction factor in degrees to reduce the likelihood
            of diagonal edges being created. Larger values reduce the likelihood of diagonal edges.
            Defaults to 15.
        allow_circles (bool, optional): If True, attempts to detect polygons that are nearly circular and
            replaces them with perfect circles. Defaults to True.
        circle_threshold (float, optional): Intersection over Union (IoU) threshold used for circle detection
            (if `allow_circles` is True). Value between 0 and 1. Defaults to 0.9.
        num_cores (int, optional): Number of CPU cores to use for parallel processing. If 1, processing is
            done sequentially. Defaults to 1.
        include_metadata (bool, optional): If True, includes metadata about the regularization process in the
            output GeoDataFrame. Defaults to False.

        **kwargs: Additional keyword arguments to pass to the `to_file` method when saving the output.

    Returns:
        gpd.GeoDataFrame: A new GeoDataFrame with regularized polygon geometries. Original attributes are
        preserved. Geometries that failed processing might be dropped.

    Raises:
        ValueError: If the input data is not a GeoDataFrame or a file path, or if the input GeoDataFrame is empty.
    """
    try:
        from buildingregulariser import regularize_geodataframe
    except ImportError:
        install_package("buildingregulariser")
        from buildingregulariser import regularize_geodataframe

    if isinstance(data, str):
        data = gpd.read_file(data)
    elif not isinstance(data, gpd.GeoDataFrame):
        raise ValueError("Input data must be a GeoDataFrame or a file path.")

    # Check if the input data is empty
    if data.empty:
        raise ValueError("Input GeoDataFrame is empty.")

    gdf = regularize_geodataframe(
        data,
        parallel_threshold=parallel_threshold,
        target_crs=target_crs,
        simplify=simplify,
        simplify_tolerance=simplify_tolerance,
        allow_45_degree=allow_45_degree,
        diagonal_threshold_reduction=diagonal_threshold_reduction,
        allow_circles=allow_circles,
        circle_threshold=circle_threshold,
        num_cores=num_cores,
        include_metadata=include_metadata,
    )

    if output_path:
        gdf.to_file(output_path, **kwargs)

    return gdf


def text_sam_gui(
    sam: SamGeo3,
    m: MapWrapper,
    overlay_bounds: MAP_BOUNDS,
    out_dir: Optional[str] = None,
    box_threshold: float = 0.25,
    text_threshold: float = 0.25,
    cmap: str = "viridis",
    opacity: float = 0.7,
    min_size: int = 10,
    max_size: Optional[int] = None,
):
    """Display the SAM Map GUI.

    Args:
        sam (SamGeo):
        m (MapWrapper): The map to display.
        out_dir (str, optional): The output directory. Defaults to None.
        box_threshold (float, optional): The threshold for the box. Defaults to 0.25.
        text_threshold (float, optional): The threshold for the text. Defaults to 0.25.
        cmap (str, optional): The colormap to use. Defaults to "viridis".
        opacity (float, optional): The opacity of the mask. Defaults to 0.7.
        min_size (int, optional): The minimum size of the object. Defaults to 10.
        max_size (int, optional): The maximum size of the object. Defaults to None.

    """
    try:
        import shutil
        import tempfile

        import ipyevents
        import ipyleaflet
        import ipywidgets as widgets
        import leafmap.colormaps as cm
        from ipyfilechooser import FileChooser
    except ImportError:
        raise ImportError(
            "The sam_map function requires the leafmap package. Please install it first."
        )

    if out_dir is None:
        out_dir = tempfile.gettempdir()

    # m = MapWrapper(**kwargs)
    m.default_style = {"cursor": "crosshair"}

    widget_width = "280px"
    button_width = "90px"
    padding = "0px 4px 0px 4px"  # upper, right, bottom, left
    style = {"description_width": "initial"}

    toolbar_button = widgets.ToggleButton(
        value=True,
        tooltip="Toolbar",
        icon="gear",
        layout=widgets.Layout(width="28px", height="28px", padding="0px 0px 0px 4px"),
    )

    close_button = widgets.ToggleButton(
        value=False,
        tooltip="Close the tool",
        icon="times",
        button_style="primary",
        layout=widgets.Layout(height="28px", width="28px", padding="0px 0px 0px 4px"),
    )

    text_prompt = widgets.Text(
        description="Text prompt:",
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    box_slider = widgets.FloatSlider(
        description="Box threshold:",
        min=0,
        max=1,
        value=box_threshold,
        step=0.01,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    if sam.model_version == "sam3":
        box_slider.description = "Conf. threshold:"

    text_slider = widgets.FloatSlider(
        description="Text threshold:",
        min=0,
        max=1,
        step=0.01,
        value=text_threshold,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    if sam.model_version == "sam3":
        text_slider.description = "Mask threshold:"

    cmap_dropdown = widgets.Dropdown(
        description="Palette:",
        options=cm.list_colormaps(),
        value=cmap,
        style=style,
        layout=widgets.Layout(width=widget_width, padding=padding),
    )

    opacity_slider = widgets.FloatSlider(
        description="Opacity:",
        min=0,
        max=1,
        value=opacity,
        readout=True,
        continuous_update=True,
        layout=widgets.Layout(width=widget_width, padding=padding),
        style=style,
    )

    def opacity_changed(change):
        if change["new"]:
            mask_layer = m.find_layer(m.layer_name)
            if mask_layer is not None:
                mask_layer.interact(opacity=opacity_slider.value)

    opacity_slider.observe(opacity_changed, "value")

    rectangular = widgets.Checkbox(
        value=False,
        description="Regularize",
        layout=widgets.Layout(width="130px", padding=padding),
        style=style,
    )

    colorpicker = widgets.ColorPicker(
        concise=False,
        description="Color",
        value="#ffff00",
        layout=widgets.Layout(width="140px", padding=padding),
        style=style,
    )

    segment_button = widgets.ToggleButton(
        description="Segment",
        value=False,
        button_style="primary",
        layout=widgets.Layout(padding=padding),
    )

    save_button = widgets.ToggleButton(
        description="Save", value=False, button_style="primary"
    )

    reset_button = widgets.ToggleButton(
        description="Reset", value=False, button_style="primary"
    )
    segment_button.layout.width = button_width
    save_button.layout.width = button_width
    reset_button.layout.width = button_width

    output = widgets.Output(
        layout=widgets.Layout(
            width=widget_width, padding=padding, max_width=widget_width
        )
    )

    toolbar_header = widgets.HBox()
    toolbar_header.children = [close_button, toolbar_button]
    toolbar_footer = widgets.VBox()
    toolbar_footer.children = [
        text_prompt,
        box_slider,
        text_slider,
        cmap_dropdown,
        opacity_slider,
        widgets.HBox([rectangular, colorpicker]),
        widgets.HBox(
            [segment_button, save_button, reset_button],
            layout=widgets.Layout(padding="0px 4px 0px 4px"),
        ),
        output,
    ]
    toolbar_widget = widgets.VBox()
    toolbar_widget.children = [toolbar_header, toolbar_footer]

    toolbar_event = ipyevents.Event(
        source=toolbar_widget, watched_events=["mouseenter", "mouseleave"]
    )

    def handle_toolbar_event(event):
        if event["type"] == "mouseenter":
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        elif event["type"] == "mouseleave":
            if not toolbar_button.value:
                toolbar_widget.children = [toolbar_button]
                toolbar_button.value = False
                close_button.value = False

    toolbar_event.on_dom_event(handle_toolbar_event)

    def toolbar_btn_click(change):
        if change["new"]:
            close_button.value = False
            toolbar_widget.children = [toolbar_header, toolbar_footer]
        else:
            if not close_button.value:
                toolbar_widget.children = [toolbar_button]

    toolbar_button.observe(toolbar_btn_click, "value")

    def close_btn_click(change):
        if change["new"]:
            toolbar_button.value = False
            if m.toolbar_control in m.controls:
                m.remove_control(m.toolbar_control)
            toolbar_widget.close()

    close_button.observe(close_btn_click, "value")

    def segment_button_click(change):
        if change["new"]:
            segment_button.value = False
            with output:
                output.clear_output()
                if len(text_prompt.value) == 0 and m.user_roi_bounds() is None:
                    print(
                        "Please enter a text prompt or draw a region of interest first."
                    )
                elif sam.source is None:
                    print("Please run sam.set_image() first.")
                else:
                    print("Segmenting...")
                    if len(text_prompt.value) > 0:
                        layer_name = text_prompt.value.replace(" ", "_")
                    elif m.user_roi_bounds() is not None:
                        layer_name = "masks"

                    filename = os.path.join(
                        out_dir, f"{layer_name}_{random_string()}.tif"
                    )
                    try:
                        if sam.model_version == "sam3":
                            sam.confidence_threshold = box_slider.value
                            sam.mask_threshold = text_slider.value
                            if len(text_prompt.value) > 0:
                                sam.generate_masks(
                                    prompt=text_prompt.value,
                                    min_size=min_size,
                                    max_size=max_size,
                                )
                            elif m.user_roi_bounds() is not None:
                                sam.generate_masks_by_boxes(
                                    boxes=[m.user_roi_bounds()],
                                    box_crs="EPSG:4326",
                                    min_size=min_size,
                                    max_size=max_size,
                                )
                            else:
                                print(
                                    "Please enter a text prompt or draw a region of interest first."
                                )
                                return
                            sam.save_masks(output=filename)
                        else:
                            print(
                                f"Unknown or unsupported model_version: {getattr(sam, 'model_version', None)}. Please set sam.model_version to 'sam2' or 'sam3'."
                            )
                            return
                        # TODO: add .output attribute to sam object
                        sam.output = filename  # type: ignore - dynamically adding attribute
                        if m.find_layer(layer_name) is not None:
                            m.remove_layer(m.find_layer(layer_name))
                        if m.find_layer(f"{layer_name}_rect") is not None:
                            m.remove_layer(m.find_layer(f"{layer_name} Regularized"))

                    except Exception as e:
                        output.clear_output()
                        print(e)
                        raise e
                    if os.path.exists(filename):
                        print("file exists, displaying the results...")
                        try:
                            print("Displaying the results on the map.")
                            add_offline_raster(
                                m=m,
                                tif_path=filename,
                                bounds=overlay_bounds,
                                opacity=opacity_slider.value,
                                palette=cmap_dropdown.value,
                                nodata=0,
                            )
                            m.layer_name = layer_name
                            print("Displayed the results on the map.")
                            if rectangular.value:
                                print("Regularizing the vector...")
                                vector = filename.replace(".tif", ".gpkg")
                                vector_rec = filename.replace(".tif", "_rect.gpkg")
                                raster_to_vector(filename, vector)
                                regularize(vector, vector_rec)
                                vector_style = {"color": colorpicker.value}
                                m.add_vector(
                                    vector_rec,
                                    layer_name=f"{layer_name} Regularized",
                                    style=vector_style,
                                    info_mode=None,
                                    zoom_to_layer=False,
                                )

                                print("saved regularized vector.")
                            print("Clearing output.")
                            output.clear_output()
                            print("Cleared output.")

                            if sam.model_version == "sam3":
                                if sam.masks is not None:
                                    print(f"Found {len(sam.masks)} objects.")
                                else:
                                    print("No objects found.")
                        except Exception as e:
                            logging.error(e)
                            print(e)
                            with open("sam_map_error.log", "a") as log_file:
                                log_file.write(f"{e}\n")
                            raise e
                    else:
                        print("Segmentation failed. No output file generated.")

    segment_button.observe(segment_button_click, "value")

    def filechooser_callback(chooser):
        with output:
            if chooser.selected is not None:
                try:
                    filename = chooser.selected
                    shutil.copy(sam.output, filename)  # type: ignore - dynamically added attribute
                    vector = filename.replace(".tif", ".gpkg")
                    raster_to_vector(filename, vector)
                    if rectangular.value:
                        vector_rec = filename.replace(".tif", "_rect.gpkg")
                        regularize(vector, vector_rec)
                except Exception as e:
                    print(e)

                if m.save_control in m.controls:
                    m.remove_control(m.save_control)
                    delattr(m, "save_control")
                save_button.value = False

    def save_button_click(change):
        if change["new"]:
            with output:
                output.clear_output()
                if not hasattr(m, "layer_name"):
                    print("Please click the Segment button first.")
                else:
                    sandbox_path = os.environ.get("SANDBOX_PATH")
                    filechooser = FileChooser(
                        path=os.getcwd(),
                        filename=f"{m.layer_name}.tif",
                        sandbox_path=sandbox_path,
                        layout=widgets.Layout(width="454px"),
                    )
                    filechooser.use_dir_icons = True
                    filechooser.filter_pattern = ["*.tif"]
                    filechooser.register_callback(filechooser_callback)
                    save_control = ipyleaflet.WidgetControl(
                        widget=filechooser, position="topright"
                    )
                    m.add_control(save_control)
                    m.save_control = save_control

        else:
            if m.save_control in m.controls:
                m.remove_control(m.save_control)
                delattr(m, "save_control")

    save_button.observe(save_button_click, "value")

    def reset_button_click(change):
        if change["new"]:
            segment_button.value = False
            save_button.value = False
            reset_button.value = False
            opacity_slider.value = 0.7
            model_version = getattr(sam, "model_version", "sam2")
            if model_version == "sam2":
                box_slider.value = 0.25
                text_slider.value = 0.25
            elif model_version == "sam3":
                box_slider.value = 0.5
                text_slider.value = 0.5
            cmap_dropdown.value = "viridis"
            text_prompt.value = ""
            output.clear_output()
            try:
                if m.find_layer(m.layer_name) is not None:
                    m.remove_layer(m.find_layer(m.layer_name))
                m.clear_drawings()
            except:
                pass

    reset_button.observe(reset_button_click, "value")

    toolbar_control = ipyleaflet.WidgetControl(
        widget=toolbar_widget, position="topright"
    )
    m.add_control(toolbar_control)
    m.toolbar_control = toolbar_control

    return m

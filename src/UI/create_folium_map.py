import logging
from pathlib import Path
from typing import Optional, Tuple, Union
import folium
import base64
from PIL import Image
import io
import webbrowser
import os


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
    # Resize if too large
    img = resize_if_too_large(img, max_dimension)
    return convert_to_base64(img)


MAP_BOUNDS = Tuple[Tuple[float, float], Tuple[float, float]]


def create_base_map(
    base_map_path: path_or_str, base_map_bounds: MAP_BOUNDS, max_dimension: int
) -> folium.Map:
    """Create a folium Map with a base map image only."""

    # Resize and encode base map image
    base_map_uri = resize_and_encode(base_map_path, max_dimension=max_dimension)

    # Calculate center point from base map bounds
    center_lat = (base_map_bounds[0][0] + base_map_bounds[1][0]) / 2
    center_lon = (base_map_bounds[0][1] + base_map_bounds[1][1]) / 2

    # Create an offline map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=10,
        tiles=None,
        attr="",
        prefer_canvas=True,
    )

    # Add base map image (bottom layer)
    folium.raster_layers.ImageOverlay(  # pyright: ignore[reportAttributeAccessIssue]
        image=base_map_uri,
        bounds=base_map_bounds,
        opacity=1.0,
        interactive=True,
        cross_origin=False,
        zindex=1,
        name="Base Map",
    ).add_to(m)

    # Fit map to show both layers
    m.fit_bounds(base_map_bounds)

    return m


def add_image_overlay(
    m: folium.Map,
    overlay_image_path: path_or_str,
    overlay_image_bounds: MAP_BOUNDS,
    max_dimension: int,
) -> folium.Map:
    """Add an image overlay to an existing folium Map."""

    # Resize and encode overlay image
    overlay_uri = resize_and_encode(overlay_image_path, max_dimension=max_dimension)

    # Add overlay image (top layer)
    folium.raster_layers.ImageOverlay(  # pyright: ignore[reportAttributeAccessIssue]
        image=overlay_uri,
        bounds=overlay_image_bounds,
        opacity=0.7,  # Semi-transparent so you can see base map
        interactive=True,
        cross_origin=False,
        zindex=2,
        name="Overlay",
    ).add_to(m)

    return m


def open_map_in_browser(output_file: Optional[path_or_str]):
    """Open the saved map HTML file in the default web browser."""

    if output_file is None:
        import tempfile

        with tempfile.NamedTemporaryFile(suffix=".html", delete=False) as temp_file:
            m.save(temp_file.name)
            open_map_in_browser(temp_file.name)
    else:
        abs_path = os.path.abspath(output_file)
        if os.name == "nt":  # Windows
            file_url = abs_path
        else:  # Unix/Linux/Mac
            file_url = f"file://{abs_path}"

        webbrowser.open(file_url, new=2)
        logging.info(f"Map opened in browser!")
        logging.info(f"File location: {abs_path}")


def create_layered_map(
    base_map_path: path_or_str,
    base_map_bounds: MAP_BOUNDS,
    overlay_image_path: path_or_str,
    overlay_image_bounds: MAP_BOUNDS,
    output_file: Optional[path_or_str] = None,
    should_open_in_browser: bool = False,
    max_dimension: int = MAX_DIMENSION,
) -> folium.Map:

    m = create_base_map(
        base_map_path=base_map_path,
        base_map_bounds=base_map_bounds,
        max_dimension=max_dimension,
    )

    m = add_image_overlay(
        m,
        overlay_image_path=overlay_image_path,
        overlay_image_bounds=overlay_image_bounds,
        max_dimension=max_dimension,
    )

    # Add layer control to toggle layers on/off
    folium.LayerControl().add_to(m)

    # Save the map
    if output_file is not None:
        m.save(output_file)
        logging.info(f"\nMap saved to: {output_file}")
    if should_open_in_browser:
        open_map_in_browser(output_file)
    return m


if __name__ == "__main__":

    map_bounds = (
        (31.550854962060072, 34.46861743927003),
        (31.572795239267688, 34.51196193695069),
    )

    overlay_bounds = (
        (31.559083170805593, 34.484871625900276),
        (31.564568240127038, 34.49570775032044),
    )

    m = create_layered_map(
        base_map_path=Path("data/example/netivot.png"),
        base_map_bounds=map_bounds,
        overlay_image_path=Path("data/example/netivot1.png"),
        overlay_image_bounds=overlay_bounds,
    )

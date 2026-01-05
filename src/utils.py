from datetime import datetime
import rasterio
from rasterio.transform import from_bounds
from typing import Union, Optional, List
from pathlib import Path
import numpy as np
from PIL import Image
import random
import string
import subprocess

MAP_BOUNDS = List[List[float]]


def image_to_geotiff(
    img_path: Union[str, Path],
    bounds: MAP_BOUNDS,
    out_path: Optional[Union[str, Path]] = None,
) -> Path:
    """
    Converts a standard image to a GeoTIFF using provided geographic bounds.
    bounds: [[south, west], [north, east]] as provided by ipyleaflet
    """
    img_path = Path(img_path)
    if out_path is None:
        out_path = img_path.with_suffix(".tif")
    else:
        out_path = Path(out_path)

    with Image.open(img_path) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        data = np.array(img)

        if len(data.shape) == 3:
            data = data.transpose(2, 0, 1)
            count = data.shape[0]
        else:
            data = data[np.newaxis, :, :]
            count = 1

    (south, west), (north, east) = bounds
    transform = from_bounds(west, south, east, north, data.shape[2], data.shape[1])

    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=data.shape[1],
        width=data.shape[2],
        count=count,
        dtype=data.dtype,
        crs="EPSG:4326",
        transform=transform,
    ) as dst:
        dst.write(data)

    return out_path


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


def write_logs(msg: str, log_file: Union[str, Path] = "sam_gui.log"):
    log_file = Path(log_file)
    log_msg = f"[{datetime.now().isoformat()}] {msg}"
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")

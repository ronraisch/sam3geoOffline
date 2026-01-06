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

from pathlib import Path
from typing import Union, Optional, Tuple
import numpy as np
from PIL import Image
import rasterio
from rasterio.control import GroundControlPoint
from rasterio.transform import from_gcps
from shapely.geometry import Polygon


def load_image_data(img_path: Path) -> Tuple[np.ndarray, int, int, int]:
    """Loads image and returns (data, count, height, width) in rasterio format."""
    with Image.open(img_path) as img:
        if img.mode not in ("RGB", "RGBA", "L"):
            img = img.convert("RGB")
        data = np.array(img)

    # Reshape to (bands, height, width)
    if data.ndim == 3:
        data = data.transpose(2, 0, 1)
    else:
        data = data[np.newaxis, :, :]
    count, height, width = data.shape
    return data, count, height, width


def get_affine_from_polygon(
    polygon: Polygon, height: int, width: int
) -> rasterio.transform.Affine:  # pyright: ignore[reportAttributeAccessIssue]
    """
    Maps polygon vertices to image corners to create a transform.
    Assumes polygon vertices follow: Top-Left, Top-Right, Bottom-Right, Bottom-Left.
    """
    # exterior.coords includes the 'closing' point (5 points for a quad)
    coords = list(polygon.exterior.coords)

    if len(coords) < 4:
        raise ValueError(
            "Polygon must have at least 4 vertices to map to image corners."
        )

    # Define Ground Control Points (row, col, lon, lat)
    # Mapping corners: (0,0), (0,W), (H,W), (H,0)
    gcps = [
        GroundControlPoint(0, 0, coords[0][0], coords[0][1]),  # Top-Left
        GroundControlPoint(0, width, coords[1][0], coords[1][1]),  # Top-Right
        GroundControlPoint(height, width, coords[2][0], coords[2][1]),  # Bottom-Right
        GroundControlPoint(height, 0, coords[3][0], coords[3][1]),  # Bottom-Left
    ]

    return from_gcps(gcps)


def image_to_geotiff(
    img_path: Union[str, Path],
    polygon: Polygon,
    out_path: Optional[Union[str, Path]] = None,
    crs: str = "EPSG:4326",
) -> Path:
    """Main pipeline to convert a grounded image to a GeoTIFF."""
    img_path = Path(img_path)
    out_path = Path(out_path) if out_path else img_path.with_suffix(".tif")

    # 1. Image Processing
    data, count, height, width = load_image_data(img_path)

    # 2. Coordinate Transformation
    # This handles non-rectangular rotations via GCPs
    transform = get_affine_from_polygon(polygon, height, width)

    # 3. Write Output
    with rasterio.open(
        out_path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=data.dtype,
        crs=crs,
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


def write_logs(msg: str, log_file: Union[str, Path] = "data/sam_gui.log"):
    log_file = Path(log_file)
    log_msg = f"[{datetime.now().isoformat()}] {msg}"
    with open(log_file, "a") as f:
        f.write(log_msg + "\n")

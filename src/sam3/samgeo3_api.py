import os
import zipfile
import shutil
import uuid
from typing import List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel
from PIL import Image
from fastapi.middleware.cors import CORSMiddleware

# Importing the SamGeo3 class
from samgeo3 import SamGeo3

app = FastAPI(title="SAM3 Geospatial & Pixel API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # This allows the Google sandbox to talk to your local server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Setup directories
UPLOAD_DIR = "uploads"
OUTPUT_DIR = "outputs"
MASKS_DIR = "masks_cache"
FOLDERS_TO_CLEAN = [UPLOAD_DIR, OUTPUT_DIR, MASKS_DIR]

for folder in FOLDERS_TO_CLEAN:
    os.makedirs(folder, exist_ok=True)

# Initialize SAM3
sam = SamGeo3(confidence_threshold=0.1, mask_threshold=0.1)

# Global state to keep track of the "current" image for the session
current_image_state = {
    "path": None,
    "masks": None,  # List of numpy arrays
    "scores": None,
}


class BoxRequest(BaseModel):
    boxes: List[List[float]]  # [[x1, y1, x2, y2], ...]


class ImagePathRequest(BaseModel):
    path: str


@app.post("/upload-path")
async def upload_from_path(request: ImagePathRequest):
    """
    Route 1: Receives a local path or URL, saves it, and returns the internal path.
    """
    input_path = request.path
    file_ext = os.path.splitext(input_path)[1] or ".png"
    unique_filename = f"{uuid.uuid4()}{file_ext}"
    dest_path = os.path.join(UPLOAD_DIR, unique_filename)

    try:
        if input_path.startswith(("http://", "https://")):
            from samgeo.common import download_file

            temp_file = download_file(input_path)
            shutil.move(temp_file, dest_path)
        else:
            if not os.path.exists(input_path):
                raise HTTPException(
                    status_code=404, detail="Input path does not exist."
                )
            shutil.copy(input_path, dest_path)

        current_image_state["path"] = dest_path
        sam.set_image(dest_path)

        return {"status": "success", "internal_path": dest_path}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict-boxes")
async def predict_boxes(request: BoxRequest):
    """
    Route 2: Runs SAM3 using pixel boxes.
    Returns a grayscale image representing the sum of masks (0 to 1).
    """
    img_path = current_image_state["path"]
    if not img_path:
        raise HTTPException(
            status_code=400, detail="No image loaded. Call /upload-path first."
        )

    try:
        results = sam.predict_batch(
            images=[img_path], input_boxes=[request.boxes], batch_size=1
        )

        img_id = list(results.keys())[0]
        masks = results[img_id]["masks"]
        scores = results[img_id]["scores"]

        current_image_state["masks"] = masks
        current_image_state["scores"] = scores

        mask_file_name = f"masks_{uuid.uuid4()}.npy"
        mask_save_path = os.path.join(MASKS_DIR, mask_file_name)
        np.save(mask_save_path, np.array(masks))

        if not masks:
            raise HTTPException(status_code=404, detail="No masks generated.")
        combined_sum = np.sum(masks, axis=0).astype(np.float32)
        max_val = combined_sum.max()
        grayscale_img = combined_sum / max_val if max_val > 0 else combined_sum

        output_img_name = f"sum_{uuid.uuid4()}.png"
        output_img_path = os.path.join(OUTPUT_DIR, output_img_name)

        # Using Pillow to save the sum image
        pil_img = Image.fromarray((grayscale_img * 255).astype(np.uint8), mode="L")
        pil_img.save(output_img_path)

        return {
            "sum_image_url": f"{OUTPUT_DIR}/{output_img_name}",
            "masks_npy_path": mask_save_path,
            "mask_count": len(masks),
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/export-geodata")
async def export_geodata(threshold: float = 0.5):
    """
    Route 3: Generates vector data from cached masks.
    Supports both GeoTIFF (geographic coords) and standard images (pixel coords).
    """
    if current_image_state["masks"] is None:
        raise HTTPException(
            status_code=400, detail="No masks found. Run /predict-boxes first."
        )

    img_path = current_image_state["path"]
    is_geospatial = img_path.lower().endswith((".tif", ".tiff"))

    try:
        # Build results for conversion
        filtered_masks = []
        filtered_scores = []
        for m, s in zip(current_image_state["masks"], current_image_state["scores"]):
            if s >= threshold:
                filtered_masks.append(m)
                filtered_scores.append(s)

        if not filtered_masks:
            return JSONResponse(
                {"message": "No polygons found above threshold."}, status_code=404
            )

        # If it's a normal image, we manually create a GDF in pixel space
        if not is_geospatial:
            import geopandas as gpd
            from shapely.geometry import shape
            from rasterio import features, transform

            # Identity transform keeps coordinates as raw pixels
            ident_transform = transform.IDENTITY
            geoms = []
            final_scores = []

            for mask, score in zip(filtered_masks, filtered_scores):
                mask_uint8 = (mask > 0).astype(np.uint8)
                # Generate shapes from mask
                for g, v in features.shapes(
                    mask_uint8, mask=mask_uint8, transform=ident_transform
                ):
                    geoms.append(shape(g))
                    final_scores.append(score)  # Assign score to every polygon part

            gdf = gpd.GeoDataFrame({"geometry": geoms, "score": final_scores}, crs=None)
        else:
            # Use the built-in geospatial logic for GeoTIFFs
            mock_results = {
                "session": {
                    "masks": filtered_masks,
                    "scores": filtered_scores,
                    "source": img_path,
                }
            }
            gdf = sam.results_to_gdf(mock_results)

        if gdf is None or gdf.empty:
            return JSONResponse(
                {"message": "Failed to vectorize masks."}, status_code=404
            )

        temp_export_id = str(uuid.uuid4())
        shape_dir = os.path.join(OUTPUT_DIR, temp_export_id)
        os.makedirs(shape_dir, exist_ok=True)

        base_name = "export_results"
        shp_path = os.path.join(shape_dir, f"{base_name}.shp")
        gdf.to_file(shp_path)

        zip_path = os.path.join(OUTPUT_DIR, f"{temp_export_id}.zip")
        with zipfile.ZipFile(zip_path, "w") as zipf:
            for root, _, files in os.walk(shape_dir):
                for file in files:
                    zipf.write(os.path.join(root, file), file)

        shutil.rmtree(shape_dir)
        return FileResponse(
            zip_path, media_type="application/zip", filename=f"export_{threshold}.zip"
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/cleanup")
async def cleanup_folders():
    """
    Route 4: Empties all related folders (uploads, outputs, masks_cache)
    and resets the current session state.
    """
    report = {}
    try:
        for folder in FOLDERS_TO_CLEAN:
            count = 0
            for filename in os.listdir(folder):
                file_path = os.path.join(folder, filename)
                try:
                    if os.path.isfile(file_path) or os.path.islink(file_path):
                        os.unlink(file_path)
                        count += 1
                    elif os.path.isdir(file_path):
                        shutil.rmtree(file_path)
                        count += 1
                except Exception as e:
                    print(f"Failed to delete {file_path}. Reason: {e}")
            report[folder] = f"Deleted {count} items"

        # Reset session state
        current_image_state["path"] = None
        current_image_state["masks"] = None
        current_image_state["scores"] = None

        return {"status": "success", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/download/{filename:path}")
async def download_file(filename: str):
    """
    Helper to serve files. Supports both raw filenames and paths with prefixes.
    e.g. /download/sum_abc.png OR /download/uploads/image_123.png
    """
    # If input is 'uploads/myfile.png', this handles both 'uploads' and 'myfile.png'
    # Normalize the path to prevent directory traversal
    clean_filename = os.path.normpath(filename).replace("..", "")

    # Check if the user provided the path directly (e.g., /download/uploads/file.png)
    # We check if it exists exactly as requested relative to current dir
    if os.path.exists(clean_filename):
        # Safety check: only allow downloads from our specific directories
        if any(
            clean_filename.startswith(folder) for folder in [UPLOAD_DIR, OUTPUT_DIR]
        ):
            return FileResponse(clean_filename)

    # Fallback: check directories manually for raw filenames
    # (e.g. user calls /download/file.png without prefix)
    output_path = os.path.join(OUTPUT_DIR, clean_filename)
    if os.path.exists(output_path):
        return FileResponse(output_path)

    upload_path = os.path.join(UPLOAD_DIR, clean_filename)
    if os.path.exists(upload_path):
        return FileResponse(upload_path)

    raise HTTPException(status_code=404, detail="File not found.")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

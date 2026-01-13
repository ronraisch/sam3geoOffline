"""Segmenting remote sensing images with the Segment Anything Model 3 (SAM3).
Optimized for batch processing, modularity, and efficient geospatial conversion.
"""

import os
import gc
import hashlib
from typing import Dict, List, Optional, Tuple, Union, TypedDict, Any

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio import features
from rasterio.io import DatasetReader
from shapely.geometry import shape, Polygon
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from src.sam3.sam3dataset import SAM3Dataset, collate_fn

from transformers import (
    Sam3Model,
    Sam3Processor,
)



try:
    import geopandas as gpd

    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from samgeo import common


class PredictionResult(TypedDict):
    """Structured container for prediction outputs."""

    masks: List[np.ndarray]
    boxes: List[np.ndarray]
    scores: List[float]
    source: Optional[str]


class SamGeo3:
    """The main class for segmenting geospatial data with SAM3."""

    def __init__(
        self,
        model_id: str = "data/sam3",
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        mask_threshold: float = 0.25,
    ) -> None:


        device_str = device or str(common.get_device())
        self.device = torch.device(device_str)
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold

        model = Sam3Model.from_pretrained(model_id)
        self.model: Sam3Model = model.to(self.device)  # type: ignore[call-arg]
        self.processor: Sam3Processor = Sam3Processor.from_pretrained(model_id)

        # Session State
        self.masks: Optional[List[np.ndarray]] = None
        self.boxes: Optional[List[np.ndarray]] = None
        self.scores: Optional[List[float]] = None
        self.source: Optional[str] = None
        self.pil_image: Optional[Image.Image] = None

    # --- Image Handling ---

    def set_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        bands: Optional[List[int]] = None,
    ) -> None:
        """Sets the current image for single-image operations."""
        self.source = image_input if isinstance(image_input, str) else None
        self.pil_image = self._to_pil(image_input, bands=bands)
        if self.pil_image is None:
            raise ValueError("Failed to load or convert image input.")

    def _to_pil(
        self,
        img_input: Union[np.ndarray, Image.Image, str],
        bands: Optional[List[int]] = None,
    ) -> Image.Image:
        """Converts various inputs to a standard RGB PIL Image."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input).convert("RGB")

        if isinstance(img_input, str):
            return self._load_from_path(img_input, bands)

        raise TypeError(f"Unsupported image type: {type(img_input)}")

    def _load_from_path(
        self, path: str, bands: Optional[List[int]] = None
    ) -> Image.Image:
        """Internal helper for loading images from local or remote paths."""
        if path.startswith("http"):
            path = str(common.download_file(path))

        if path.lower().endswith((".tif", ".tiff")):
            try:
                with rasterio.open(path) as src:
                    arr = src.read(bands) if bands else src.read()[:3]
                    arr = arr.astype(np.float32)
                    arr = (arr - arr.min()) / (arr.max() - arr.min() + 1e-6)
                    return Image.fromarray(
                        (arr.transpose(1, 2, 0) * 255).astype(np.uint8)
                    )
            except Exception:
                # Fallback for standard images if rasterio fails
                pass

        img = cv2.imread(path)
        if img is None:
            raise IOError(f"Could not read image at {path}")
        return Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

    def _get_image_id(self, img: Any) -> str:
        """Returns filename if string, or hash if image data."""
        if isinstance(img, str):
            return os.path.basename(img)
        data = np.array(img).tobytes()
        return hashlib.md5(data).hexdigest()[:12]

    # --- Core Inference Engine ---

    def _execute_batch(
        self,
        batch_imgs: List[Any],
        processor_kwargs: Dict[str, Any],
        clear_vram: bool = False,
    ) -> List[PredictionResult]:
        """Handles the heavy lifting of a single batch forward pass."""
        pil_imgs = [self._to_pil(img) for img in batch_imgs]
        target_sizes = [[img.height, img.width] for img in pil_imgs]

        # SAM3 requires a text prompt (or empty string) even for box-only prompts
        if "text" not in processor_kwargs:
            processor_kwargs["text"] = [""] * len(batch_imgs)

        inputs = self.processor(
            images=pil_imgs, **processor_kwargs, return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        raw_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )

        formatted = self._format_batch_results(raw_results, batch_imgs)

        del inputs, outputs, raw_results
        self._free_vram(full_cleanup=clear_vram)
        return formatted

    def _format_batch_results(
        self, raw: List[Dict], original_inputs: List[Any]
    ) -> List[PredictionResult]:
        """Converts raw torch output to PredictionResult list."""
        results = []
        for i, res in enumerate(raw):
            results.append(
                {
                    "masks": [m.cpu().numpy() for m in res["masks"]],
                    "boxes": [b.cpu().numpy() for b in res["boxes"]],
                    "scores": [float(s.cpu().item()) for s in res["scores"]],
                    "source": (
                        original_inputs[i]
                        if isinstance(original_inputs[i], str)
                        else None
                    ),
                }
            )
        return results

    # --- Public Prediction Methods ---

    def predict_image_boxes(
        self, image: Union[np.ndarray, Image.Image, str], boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Predicts masks for a single image given a list of bounding boxes in pixel coords.

        Args:
            image: Image input (path, array, or PIL).
            boxes: List of boxes, each as [x1, y1, x2, y2] in pixel coordinates.

        Returns:
            List of binary numpy masks.
        """
        # Format for batch prediction: [image_idx][box_idx][coords]
        input_boxes = [boxes]
        # convert to float

        results = self.predict_batch(
            images=[image], input_boxes=input_boxes, batch_size=1
        )

        # Extract masks from the single result
        image_id = self._get_image_id(image)
        return results[image_id]["masks"]

    def predict_batch(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        prompts: Optional[Union[str, List[str]]] = None,
        input_boxes: Optional[List[List[List[float]]]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        verbose: bool = True,
        clear_vram: bool = False,
    ) -> Dict[str, PredictionResult]:
        """
        Generic batch prediction method supporting text or boxes.

        Args:
            images: List of images (path, array, or PIL).
            prompts: Text prompt(s).
            input_boxes: List of bounding boxes per image [[[x1, y1, x2, y2]]].
        """
        all_results: Dict[str, PredictionResult] = {}
        
        dataset = SAM3Dataset(
            images=images, 
            prompts=prompts, 
            input_boxes=input_boxes, 
            id_func=self._get_image_id
        )
        
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )

        for batch in tqdm(dataloader, desc="SAM3 Batch", disable=not verbose):
            batch_imgs = batch["image"]
            batch_ids = batch["id"]

            kwargs = {}
            if "prompt" in batch:
                kwargs["text"] = batch["prompt"]

            if "boxes" in batch:
                kwargs["input_boxes"] = batch["boxes"]

            # Execute prediction
            batch_out = self._execute_batch(batch_imgs, kwargs, clear_vram=clear_vram)
            
            for img_id, res in zip(batch_ids, batch_out):
                all_results[img_id] = res

        return all_results
    # --- Geospatial Processing ---

    def results_to_gdf(
        self, results: Dict[str, PredictionResult]
    ) -> Optional[gpd.GeoDataFrame]:
        """Converts prediction dictionary into a GeoDataFrame with geographic coordinates."""
        if not HAS_GEOSPATIAL_LIBS:
            raise ImportError("geopandas and shapely are required.")

        data = {"geometry": [], "image_id": [], "score": []}
        common_crs = None

        for _, res in results.items():
            src_path = res.get("source")
            if not (src_path and src_path.lower().endswith((".tif", ".tiff"))):
                continue

            try:
                transform, crs = self._get_geospatial_metadata(src_path)
                if common_crs is None:
                    common_crs = crs

                for mask, score in zip(res["masks"], res["scores"]):
                    geoms = self._mask_to_polygons(mask, transform)
                    for poly in geoms:
                        data["geometry"].append(poly)
                        data["image_id"].append(src_path)
                        data["score"].append(score)
            except Exception:
                continue
        # TODO: return empty geodataframe
        return gpd.GeoDataFrame(data, crs=common_crs) if data["geometry"] else None

    def _get_geospatial_metadata(self, path: str) -> Tuple[Any, Any]:
        """Extracts transform and CRS from a GeoTIFF."""
        with rasterio.open(path) as src:
            return src.transform, src.crs

    def _mask_to_polygons(self, mask: np.ndarray, transform: Any) -> List[Polygon]:
        """Converts a single binary mask into a list of polygons."""
        if not np.any(mask):
            return []

        polygons = []
        mask_uint8 = mask.astype(np.uint8)
        if mask_uint8.max() == 1:
            mask_uint8 *= 255

        for geom, _ in features.shapes(
            mask_uint8, mask=(mask_uint8 > 0), transform=transform
        ):
            poly = shape(geom)
            if not poly.is_empty:
                polygons.append(poly)
        return polygons

    # --- Utility ---

    def _free_vram(self, full_cleanup: bool = False) -> None:
        """Releases memory from device."""
        if full_cleanup:
            self.model.cpu()
        gc.collect()
        if "cuda" in str(self.device):
            torch.cuda.empty_cache()
        if full_cleanup:
            self.model.to(self.device)  # type: ignore[call-arg]

    # --- Legacy Wrappers ---

    def generate_masks(self, prompt: str) -> None:
        """Sets internal state using single image text prompt."""
        if not self.pil_image:
            raise ValueError("Set image first.")
        res = list(
            self.predict_batch([self.source or self.pil_image], prompts=prompt).values()
        )[0]
        self.masks, self.boxes, self.scores = res["masks"], res["boxes"], res["scores"]


if __name__ == "__main__":
    sam = SamGeo3()
    # Path to a normal image
    img_path = "data/example/image.png"

    # --- EXAMPLE 1: Text Prompt ---
    print("\n--- Running Example 1: Text Prompt ---")
    text_results = sam.predict_batch(
        images=[img_path], prompts="the red car", batch_size=1
    )
    print(f"Text Prompt Example: Found {len(text_results)} image(s) in result batch.")

    # --- EXAMPLE 2: predict_image_boxes (Pixel Coordinates) ---
    print("\n--- Running Example 2: predict_image_boxes ---")

    # Bounding boxes in pixel coordinates [x1, y1, x2, y2]
    # Representing specific regions of interest in the image
    pixel_boxes = [[100.0, 150.0, 300.0, 400.0], [450.0, 50.0, 600.0, 200.0]]

    masks = sam.predict_image_boxes(image=img_path, boxes=pixel_boxes)

    print(f"predict_image_boxes Example: Generated {len(masks)} mask(s).")

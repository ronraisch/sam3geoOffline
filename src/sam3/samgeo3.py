"""Segmenting remote sensing images with the Segment Anything Model 3 (SAM3).
Optimized for batch processing, prompt sharing, and visual prompts with strict type hinting.
"""

import os
import gc
import hashlib
from typing import Dict, List, Optional, Tuple, Union, TypedDict

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio import features
from rasterio.io import DatasetReader
import torch
from tqdm import tqdm

try:
    from transformers import (
        Sam3Model,  # pyright: ignore[reportAttributeAccessIssue, reportMissingImports]
        Sam3Processor as TransformersSam3Processor,  # pyright: ignore[reportAttributeAccessIssue, reportMissingImports]
    )

    SAM3_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SAM3_TRANSFORMERS_AVAILABLE = False

try:
    import geopandas as gpd
    from shapely.geometry import shape

    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from samgeo import common


class PredictionResult(TypedDict):
    """Structured container for prediction outputs."""

    masks: List[np.ndarray]
    boxes: List[np.ndarray]
    scores: List[float]
    source: Optional[str]  # Tracks the original file path or ID for metadata


class SamGeo3:
    """The main class for segmenting geospatial data with SAM3 (Transformers backend)."""

    def __init__(
        self,
        model_id: str = "data/sam3",
        device: Optional[str] = None,
        confidence_threshold: float = 0.25,
        mask_threshold: float = 0.25,
    ) -> None:
        if not SAM3_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers SAM3 is not available. Please install it as:\n\tpip install transformers torch"
            )

        if device is None:
            device = str(common.get_device())

        self.device: str = device
        self.confidence_threshold: float = confidence_threshold
        self.mask_threshold: float = mask_threshold
        self.model_id: str = model_id

        # Initialize Backend
        self.model: Sam3Model = Sam3Model.from_pretrained(model_id).to(device)
        self.processor: TransformersSam3Processor = (
            TransformersSam3Processor.from_pretrained(model_id)
        )

        # State management
        self.masks: Optional[List[np.ndarray]] = None
        self.boxes: Optional[List[np.ndarray]] = None
        self.scores: Optional[List[float]] = None
        self.image: Optional[np.ndarray] = None
        self.source: Optional[str] = None
        self.image_height: Optional[int] = None
        self.image_width: Optional[int] = None
        self.pil_image: Optional[Image.Image] = None

    def _get_image_id(self, img: Union[np.ndarray, Image.Image, str]) -> str:
        """Generates a unique string identifier for an image input."""
        if isinstance(img, str):
            return os.path.basename(img)

        # Hash pixel data for non-path inputs
        if isinstance(img, Image.Image):
            data = np.array(img).tobytes()
        else:
            data = img.tobytes()
        return hashlib.md5(data).hexdigest()[:12]

    def set_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        bands: Optional[List[int]] = None,
    ) -> None:
        """Set the current image and prepare for inference."""
        if isinstance(image_input, str):
            self._set_image_with_string(image_input, bands=bands)
        elif isinstance(image_input, np.ndarray):
            self.image = image_input
            self.source = None
        elif isinstance(image_input, Image.Image):
            self.image = np.array(image_input)
            self.source = None

        if self.image is None:
            raise ValueError("Failed to load image.")

        self.image_height, self.image_width = self.image.shape[:2]
        self.pil_image = Image.fromarray(self.image)

    def _set_image_with_string(
        self, image_path: str, bands: Optional[List[int]] = None
    ) -> None:
        """Internal helper for loading images from path."""
        if image_path.startswith("http"):
            image_path = str(common.download_file(image_path))

        if not os.path.exists(image_path):
            raise ValueError(f"Path {image_path} does not exist.")

        self.source = image_path
        if image_path.lower().endswith((".tif", ".tiff")):
            with rasterio.open(image_path) as src:
                src: DatasetReader
                if bands is not None:
                    array: np.ndarray = np.stack([src.read(b) for b in bands], axis=0)
                else:
                    array: np.ndarray = (
                        src.read()[:3, :, :]
                        if src.count >= 3
                        else np.repeat(src.read(1)[None, :, :], 3, axis=0)
                    )

                # Normalize and transpose
                array_f: np.ndarray = array.astype(np.float32)
                array_f -= array_f.min()
                max_val: float = float(array_f.max())
                if max_val > 0:
                    array_f /= max_val
                self.image = (array_f.transpose(1, 2, 0) * 255).astype(np.uint8)
        else:
            loaded_img: Optional[np.ndarray] = cv2.imread(image_path)
            if loaded_img is None:
                raise ValueError(f"CV2 failed to load {image_path}")
            self.image = cv2.cvtColor(loaded_img, cv2.COLOR_BGR2RGB)

    def _free_vram(self, full_cleanup: bool = False) -> None:
        """Explicitly clear VRAM to avoid memory leaks."""
        if full_cleanup:
            self.model.cpu()
            self.masks = None
            self.boxes = None
            self.scores = None

        gc.collect()

        if "cuda" in self.device:
            torch.cuda.empty_cache()
            torch.cuda.ipc_collect()
        elif "mps" in self.device:
            torch.mps.empty_cache()

        if full_cleanup:
            self.model.to(self.device)

    def predict_batch_text(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        prompts: Union[str, List[str]],
        batch_size: int = 4,
        clear_model_weights: bool = False,
    ) -> Dict[str, PredictionResult]:
        """Perform batch prediction using text prompts. Returns results keyed by image identifier."""
        num_images: int = len(images)
        if isinstance(prompts, str):
            prompts_list: List[str] = [prompts] * num_images
        else:
            prompts_list = prompts

        if len(images) != len(prompts_list):
            raise ValueError("Number of images must match number of prompts.")

        all_results: Dict[str, PredictionResult] = {}
        for i in tqdm(
            range(0, num_images, batch_size), desc="SAM3 Text Batch", unit="batch"
        ):
            batch_slice = slice(i, i + batch_size)
            batch_imgs = images[batch_slice]
            batch_prompts = prompts_list[batch_slice]
            batch_ids = [self._get_image_id(img) for img in batch_imgs]

            pil_imgs: List[Image.Image] = [self._to_pil(img) for img in batch_imgs]
            target_sizes: List[List[int]] = [
                [img.height, img.width] for img in pil_imgs
            ]

            inputs = self.processor(
                images=pil_imgs, text=batch_prompts, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_out: List[Dict[str, torch.Tensor]] = (
                self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.confidence_threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=target_sizes,
                )
            )

            formatted = self._format_results(batch_out)
            for j, (img_id, res) in enumerate(zip(batch_ids, formatted)):
                # Attach source path if it exists for geospatial metadata
                original_input = batch_imgs[j]
                res["source"] = (
                    original_input if isinstance(original_input, str) else None
                )
                all_results[img_id] = res

            del inputs, outputs, batch_out
            self._free_vram(full_cleanup=clear_model_weights)

        return all_results

    def predict_batch_image_prompt(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        prompt_image: Union[np.ndarray, Image.Image, str],
        batch_size: int = 4,
        clear_model_weights: bool = False,
    ) -> Dict[str, PredictionResult]:
        """Perform batch prediction using visual prompt. Returns results keyed by image identifier."""
        num_images: int = len(images)
        pil_prompt: Image.Image = self._to_pil(prompt_image)
        all_results: Dict[str, PredictionResult] = {}

        for i in tqdm(range(0, num_images, batch_size), desc="SAM3 Visual Batch"):
            batch_slice = slice(i, i + batch_size)
            batch_imgs = images[batch_slice]
            batch_ids = [self._get_image_id(img) for img in batch_imgs]

            pil_imgs: List[Image.Image] = [self._to_pil(img) for img in batch_imgs]
            target_sizes: List[List[int]] = [
                [img.height, img.width] for img in pil_imgs
            ]
            batch_prompt_images: List[Image.Image] = [pil_prompt] * len(pil_imgs)

            inputs = self.processor(
                images=pil_imgs, prompt_images=batch_prompt_images, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_out: List[Dict[str, torch.Tensor]] = (
                self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.confidence_threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=target_sizes,
                )
            )

            formatted = self._format_results(batch_out)
            for j, (img_id, res) in enumerate(zip(batch_ids, formatted)):
                original_input = batch_imgs[j]
                res["source"] = (
                    original_input if isinstance(original_input, str) else None
                )
                all_results[img_id] = res

            del inputs, outputs, batch_out
            self._free_vram(full_cleanup=clear_model_weights)

        return all_results

    def predict_batch_boxes(
        self,
        images: List[Union[np.ndarray, Image.Image, str]],
        boxes: List[List[List[float]]],
        batch_size: int = 4,
        clear_model_weights: bool = False,
    ) -> Dict[str, PredictionResult]:
        """Perform batch prediction using bounding box prompts. Returns results keyed by image identifier."""
        if len(images) != len(boxes):
            raise ValueError("Number of images must match list of box prompts.")

        all_results: Dict[str, PredictionResult] = {}
        for i in tqdm(range(0, len(images), batch_size), desc="SAM3 Box Batch"):
            batch_slice = slice(i, i + batch_size)
            batch_imgs = images[batch_slice]
            batch_boxes = boxes[batch_slice]
            batch_ids = [self._get_image_id(img) for img in batch_imgs]

            pil_imgs: List[Image.Image] = [self._to_pil(img) for img in batch_imgs]
            target_sizes: List[List[int]] = [
                [img.height, img.width] for img in pil_imgs
            ]

            inputs = self.processor(
                images=pil_imgs, input_boxes=batch_boxes, return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                outputs = self.model(**inputs)

            batch_out: List[Dict[str, torch.Tensor]] = (
                self.processor.post_process_instance_segmentation(
                    outputs,
                    threshold=self.confidence_threshold,
                    mask_threshold=self.mask_threshold,
                    target_sizes=target_sizes,
                )
            )

            formatted = self._format_results(batch_out)
            for j, (img_id, res) in enumerate(zip(batch_ids, formatted)):
                original_input = batch_imgs[j]
                res["source"] = (
                    original_input if isinstance(original_input, str) else None
                )
                all_results[img_id] = res

            del inputs, outputs, batch_out
            self._free_vram(full_cleanup=clear_model_weights)

        return all_results

    def results_to_gdf(
        self,
        results: Dict[str, PredictionResult],
        simplify_tolerance: float = 0.0,
    ) -> Optional["gpd.GeoDataFrame"]:
        """
        Processes batch results to create a GeoDataFrame.
        For keys representing .tif files, it uses the embedded coordinate systems.
        """
        if not HAS_GEOSPATIAL_LIBS:
            raise ImportError("geopandas and shapely are required for this method.")

        all_geoms: List[shape] = []
        all_ids: List[str] = []
        all_scores: List[float] = []
        target_crs = None

        for img_id, res in results.items():
            source_path = res.get("source")

            # Skip if we don't have a file path to pull coordinates from
            if not source_path or not os.path.exists(source_path):
                continue

            # Only support .tif/.tiff for spatial metadata currently
            if not source_path.lower().endswith((".tif", ".tiff")):
                continue

            with rasterio.open(source_path) as src:
                transform = src.transform
                if target_crs is None:
                    target_crs = src.crs

            for mask, score in zip(res["masks"], res["scores"]):
                if not np.any(mask):
                    continue

                # Convert mask to uint8 for polygonization
                mask_uint8 = mask.astype(np.uint8)
                if mask_uint8.max() == 1:
                    mask_uint8 *= 255

                # Extract shapes using the image's specific transform
                extracted_shapes = features.shapes(
                    mask_uint8, mask=(mask_uint8 > 0), transform=transform
                )

                for g, v in extracted_shapes:
                    poly = shape(g)
                    if simplify_tolerance > 0:
                        poly = poly.simplify(simplify_tolerance, preserve_topology=True)

                    if not poly.is_empty:
                        all_geoms.append(poly)
                        all_ids.append(source_path)
                        all_scores.append(score)

        if not all_geoms:
            return None

        return gpd.GeoDataFrame(
            {"geometry": all_geoms, "image_id": all_ids, "score": all_scores},
            crs=target_crs,
        )

    def _to_pil(self, img_input: Union[np.ndarray, Image.Image, str]) -> Image.Image:
        """Convert any input type to PIL with explicit typing."""
        if isinstance(img_input, str):
            if img_input.lower().endswith((".tif", ".tiff")):
                with rasterio.open(img_input) as src:
                    arr: np.ndarray = src.read()[:3, :, :]
                    min_v: float = float(arr.min())
                    max_v: float = float(arr.max())
                    arr_norm: np.ndarray = (
                        (arr - min_v) / (max_v - min_v + 1e-6) * 255
                    ).astype(np.uint8)
                    return Image.fromarray(arr_norm.transpose(1, 2, 0))
            return Image.open(img_input).convert("RGB")
        if isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input)
        return img_input

    def _format_results(
        self, batch_results: List[Dict[str, torch.Tensor]]
    ) -> List[PredictionResult]:
        """Clean up tensors to numpy for backend storage."""
        formatted: List[PredictionResult] = []
        for res in batch_results:
            formatted.append(
                {
                    "masks": [m.cpu().numpy() for m in res["masks"]],
                    "boxes": [b.cpu().numpy() for b in res["boxes"]],
                    "scores": [float(s.cpu().item()) for s in res["scores"]],
                    "source": None,
                }
            )
        return formatted

    def generate_masks(self, prompt: str) -> None:
        """Legacy support for single image text prediction."""
        if self.pil_image is None:
            raise ValueError("No image set.")
        # Key will be the source name if available
        results_map = self.predict_batch_text(
            [self.source if self.source else self.pil_image], prompt
        )
        res = list(results_map.values())[0]
        self.masks, self.boxes, self.scores = res["masks"], res["boxes"], res["scores"]

    def generate_masks_by_boxes(self, boxes: List[List[float]]) -> None:
        """Legacy support for single image box prediction."""
        if self.pil_image is None:
            raise ValueError("No image set.")
        results_map = self.predict_batch_boxes(
            [self.source if self.source else self.pil_image], [boxes]
        )
        res = list(results_map.values())[0]
        self.masks, self.boxes, self.scores = res["masks"], res["boxes"], res["scores"]


if __name__ == "__main__":
    import matplotlib.pyplot as plt

    print("Initializing SamGeo3...")
    sam = SamGeo3(model_id="data/sam3")

    img_1 = "data/example/netivot1.tif"
    img_2 = "data/example/netivot2.tif"

    # Keys will be 'netivot1.tif' and 'netivot2.tif'
    text_results = sam.predict_batch_text([img_1, img_2], prompts="building")

    # Use the new results_to_gdf method
    gdf = sam.results_to_gdf(text_results, simplify_tolerance=0.1)
    if gdf is not None:
        print(f"Total features created: {len(gdf)}")

    for img_id, res in text_results.items():
        print(f"Image {img_id}: Found {len(res['masks'])} objects.")

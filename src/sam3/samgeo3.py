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
    from shapely.geometry import shape, MultiPolygon

    HAS_GEOSPATIAL_LIBS = True
except ImportError:
    HAS_GEOSPATIAL_LIBS = False

from samgeo import common


class PredictionResult(TypedDict):
    """Structured container for prediction outputs."""

    masks: List[np.ndarray]
    boxes: List[np.ndarray]
    scores: List[float]


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
        # Track mask origins for vectorization
        self.mask_origins: Optional[List[str]] = None

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
            self.mask_origins = None

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
            for img_id, res in zip(batch_ids, formatted):
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
            for img_id, res in zip(batch_ids, formatted):
                all_results[img_id] = res

            del inputs, outputs, batch_out
            self._free_vram(full_cleanup=clear_model_weights)

        return all_results

    def save_masks_as_shp(
        self, output_path: str, simplify_tolerance: float = 0.0, merge_all: bool = False
    ) -> None:
        """
        Convert generated masks into a geospatial Shapefile.
        Each feature includes an 'image_id' attribute tracing it back to the source.
        """
        if not HAS_GEOSPATIAL_LIBS:
            raise ImportError("geopandas and shapely are required for this method.")
        if self.masks is None or len(self.masks) == 0:
            raise ValueError("No masks available. Run prediction first.")
        if self.source is None:
            raise ValueError("No source image (TIF) set. CRS cannot be determined.")

        with rasterio.open(self.source) as src:
            transform = src.transform
            crs = src.crs

        geoms: List[shape] = []
        ids: List[str] = []

        # If mask_origins isn't set (legacy/single image), use the current source name
        origins = (
            self.mask_origins
            if self.mask_origins
            else [os.path.basename(self.source)] * len(self.masks)
        )

        for mask, origin_id in zip(self.masks, origins):
            mask_uint8 = mask.astype(np.uint8)
            shapes = features.shapes(
                mask_uint8, mask=(mask_uint8 > 0), transform=transform
            )

            for g, v in shapes:
                s = shape(g)
                if simplify_tolerance > 0:
                    s = s.simplify(simplify_tolerance, preserve_topology=True)
                geoms.append(s)
                ids.append(origin_id)

        if not geoms:
            raise ValueError("Vectorization resulted in no geometries.")

        if merge_all:
            # Merging by ID
            gdf_raw = gpd.GeoDataFrame({"geometry": geoms, "image_id": ids}, crs=crs)
            gdf = gdf_raw.dissolve(by="image_id").reset_index()
        else:
            gdf = gpd.GeoDataFrame({"geometry": geoms, "image_id": ids}, crs=crs)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        gdf.to_file(output_path)
        print(f"Saved {len(gdf)} features with image attributes to {output_path}")

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
                }
            )
        return formatted

    def generate_masks(self, prompt: str) -> None:
        """Legacy support for single image text prediction. Updates state for vectorization."""
        if self.pil_image is None:
            raise ValueError("No image set.")

        img_id = self._get_image_id(self.source) if self.source else "current_image"
        results_map = self.predict_batch_text([self.pil_image], prompt)
        res = results_map[list(results_map.keys())[0]]

        self.masks, self.boxes, self.scores = (
            res["masks"],
            res["boxes"],
            res["scores"],
        )
        # Store origins so save_masks_as_shp knows where they came from
        self.mask_origins = [img_id] * len(self.masks)

    def generate_masks_by_boxes(self, boxes: List[List[float]]) -> None:
        """Legacy support for single image box prediction."""
        if self.pil_image is None:
            raise ValueError("No image set.")

        # For simplicity, using a modified internal version of predict_batch_boxes
        # but maintaining the dict-based logic
        img_id = self._get_image_id(self.source) if self.source else "current_image"

        pil_imgs = [self.pil_image]
        target_sizes = [[self.pil_image.height, self.pil_image.width]]
        inputs = self.processor(
            images=pil_imgs, input_boxes=[boxes], return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        batch_out = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )

        res = self._format_results(batch_out)[0]
        self.masks, self.boxes, self.scores = res["masks"], res["boxes"], res["scores"]
        self.mask_origins = [img_id] * len(self.masks)

        del inputs, outputs, batch_out
        self._free_vram()


if __name__ == "__main__":
    # Example Usage Script
    print("Initializing SamGeo3...")
    sam = SamGeo3(model_id="data/sam3")

    img_1 = "data/example/netivot1.tif"
    img_2 = "data/example/netivot2.tif"

    # 1. Batch Prediction with Identifiers
    results = sam.predict_batch_text([img_1, img_2], "greenhouses")

    # Results is now a Dict: {'netivot1.tif': {...}, 'netivot2.tif': {...}}
    for img_id, data in results.items():
        print(f"Image {img_id}: Detected {len(data['masks'])} objects.")

    # 2. Saving to Shapefile with Attributes
    # Note: save_masks_as_shp uses self.masks, so we set state for the example
    sam.set_image(img_1)
    sam.generate_masks("swimming pool")
    sam.save_masks_as_shp("outputs/pools_with_ids.shp", simplify_tolerance=0.3)

"""Segmenting remote sensing images with the Segment Anything Model 3 (SAM3).
Optimized for batch processing, modularity, and efficient geospatial conversion.
"""

import logging
import os
import gc
import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, TypedDict, Any

import cv2
import numpy as np
from PIL import Image
import rasterio
from rasterio import features
from shapely.geometry import shape, Polygon
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import sys
if os.getcwd() not in sys.path:
    sys.path.append(os.getcwd())
from src.sam3.sam3dataset import SAM3Dataset, collate_fn

from transformers import (
    Sam3Model,
    Sam3Processor,
)

logger = logging.getLogger(__name__)



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
        confidence_threshold: float = 0.1,
        mask_threshold: float = 0.5,
    ) -> None:
        logger.info(f"Initializing SamGeo3 with model_id={model_id}, device={device}")

        device_str = device or str(common.get_device())
        self.device = torch.device(device_str)
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        logger.debug(f"Device: {self.device}, confidence_threshold: {confidence_threshold}, mask_threshold: {mask_threshold}")

        logger.info("Loading SAM3 model...")
        model = Sam3Model.from_pretrained(model_id)
        self.model: Sam3Model = model.to(self.device)  # type: ignore[call-arg]
        logger.info("Loading SAM3 processor...")
        self.processor: Sam3Processor = Sam3Processor.from_pretrained(model_id)
        logger.info("SamGeo3 initialized successfully")

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
        img_input: Union[np.ndarray, Image.Image, str, Path],
        bands: Optional[List[int]] = None,
    ) -> Image.Image:
        """Converts various inputs to a standard RGB PIL Image."""
        if isinstance(img_input, Image.Image):
            return img_input.convert("RGB")

        if isinstance(img_input, np.ndarray):
            return Image.fromarray(img_input).convert("RGB")

        if isinstance(img_input, (str, Path)):
            return self._load_from_path(str(img_input), bands)

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
        batch_imgs: List[Image.Image],
        processor_kwargs: Dict[str, Any],
        batch_sources: Optional[List[Optional[str]]] = None,
        clear_vram: bool = False,
    ) -> List[PredictionResult]:
        """
        Handles the heavy lifting of a single batch forward pass.
        
        Args:
            batch_imgs: List of PIL Images (already loaded by DataLoader)
            processor_kwargs: Arguments to pass to the processor (text, input_boxes, etc.)
            batch_sources: Optional list of source paths for each image
            clear_vram: Whether to clear VRAM after processing
        """
        logger.debug(f"Executing batch with {len(batch_imgs)} images")
        logger.debug(f"Processor kwargs keys: {list(processor_kwargs.keys())}")
        
        # Images are already PIL Images from the DataLoader
        # Store original sizes for post-processing (before processor resizing)
        target_sizes = [[img.height, img.width] for img in batch_imgs]
        logger.debug(f"Original image sizes: {target_sizes}")

        # SAM3 requires a text prompt (or empty string) even for box-only prompts
        if "text" not in processor_kwargs:
            processor_kwargs["text"] = [""] * len(batch_imgs)
            logger.debug("Added empty text prompts for box-only mode")

        logger.debug("Processing images through processor...")
        inputs = self.processor(
            images=batch_imgs, **processor_kwargs, return_tensors="pt"
        ).to(self.device)
        logger.debug(f"Inputs moved to device {self.device}")

        logger.debug("Running model forward pass...")
        with torch.no_grad():
            outputs = self.model(**inputs)
        logger.debug("Model forward pass completed")

        logger.debug("Post-processing segmentation results...")
        raw_results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=target_sizes,
        )
        logger.debug(f"Post-processing completed, got {len(raw_results)} results")

        formatted = self._format_batch_results(raw_results, batch_sources or [None] * len(batch_imgs))

        del inputs, outputs, raw_results
        self._free_vram(full_cleanup=clear_vram)
        logger.debug("Batch execution completed")
        return formatted

    def _format_batch_results(
        self, raw: List[Dict], batch_sources: List[Optional[str]]
    ) -> List[PredictionResult]:
        """Converts raw torch output to PredictionResult list."""
        results = []
        for i, res in enumerate(raw):
            results.append(
                {
                    "masks": [m.cpu().numpy() for m in res["masks"]],
                    "boxes": [b.cpu().numpy() for b in res["boxes"]],
                    "scores": [float(s.cpu().item()) for s in res["scores"]],
                    "source": batch_sources[i] if i < len(batch_sources) else None,
                }
            )
        return results

    # --- Public Prediction Methods ---

    def predict_image_boxes(
        self, image: Union[np.ndarray, Image.Image, str], boxes: List[List[float]]
    ) -> List[np.ndarray]:
        """
        Predicts masks for a single image given a list of bounding boxes in pixel coords.
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
        num_workers: Optional[int] = None,
        verbose: bool = True,
        clear_vram: bool = False,
    ) -> Dict[str, PredictionResult]:
        """
        Generic batch prediction method supporting text or boxes.

        Args:
            images: List of images (path, array, or PIL). PIL Images are converted to numpy arrays.
            prompts: Text prompt(s).
            input_boxes: List of bounding boxes per image [[[x1, y1, x2, y2]]].
        """
        logger.info(f"Starting batch prediction with {len(images)} images, batch_size={batch_size}")
        all_results: Dict[str, PredictionResult] = {}
        
        # Create mapping from image ID to source path before processing
        id_to_source: Dict[str, Optional[str]] = {}
        for img in images:
            img_id = self._get_image_id(img)
            id_to_source[img_id] = img if isinstance(img, str) else None
        logger.debug(f"Created ID to source mapping for {len(id_to_source)} images")
        
        # Convert PIL Images to numpy arrays (dataset only accepts str/Path/np.ndarray)
        dataset_images: List[Union[str, Path, np.ndarray]] = []
        for img in images:
            if isinstance(img, Image.Image):
                # Convert PIL Image to numpy array
                dataset_images.append(np.array(img))
                logger.debug(f"Converted PIL Image to numpy array")
            elif isinstance(img, (str, Path, np.ndarray)):
                dataset_images.append(img)  # type: ignore[arg-type]
            else:
                raise TypeError(f"Unsupported image type: {type(img)}")
        
        # Use num_workers=0 by default to avoid multiprocessing issues with instance methods
        # Instance methods can't be pickled, which causes problems with DataLoader workers
        num_workers = num_workers if num_workers is not None else 0
        logger.debug(f"Using num_workers={num_workers}")
        
        logger.debug("Creating SAM3Dataset...")
        dataset = SAM3Dataset(
            images=dataset_images, 
            prompts=prompts, 
            input_boxes=input_boxes, 
            id_func=self._get_image_id,
            image_loader=self._to_pil  # Pass the image loading method to the dataset
        )
        logger.debug(f"Dataset created with {len(dataset)} items")
        
        logger.debug("Creating DataLoader...")
        dataloader = DataLoader(
            dataset, 
            batch_size=batch_size, 
            shuffle=False, 
            num_workers=num_workers,
            collate_fn=collate_fn
        )
        logger.debug(f"DataLoader created, will process {len(dataloader)} batches")

        logger.info("Starting batch processing loop...")
        for batch_idx, batch in enumerate(tqdm(dataloader, desc="SAM3 Batch", disable=not verbose)):
            logger.debug(f"Processing batch {batch_idx + 1}/{len(dataloader)}")
            batch_imgs = batch["image"]  # Already PIL Images from DataLoader
            batch_ids = batch["id"]
            logger.debug(f"Batch contains {len(batch_imgs)} images with IDs: {batch_ids}")

            # Map batch IDs to source paths
            batch_sources = [id_to_source.get(img_id) for img_id in batch_ids]
            
            kwargs = {}
            if "prompt" in batch:
                kwargs["text"] = batch["prompt"]
                logger.debug(f"Batch has {len(batch['prompt'])} prompts")

            if "boxes" in batch:
                kwargs["input_boxes"] = batch["boxes"]
                logger.debug(f"Batch has {len(batch['boxes'])} box sets")

            # Execute prediction
            logger.debug(f"Executing batch {batch_idx + 1}...")
            batch_out = self._execute_batch(batch_imgs, kwargs, batch_sources=batch_sources, clear_vram=clear_vram)
            logger.debug(f"Batch {batch_idx + 1} completed, got {len(batch_out)} results")
            
            for img_id, res in zip(batch_ids, batch_out):
                all_results[img_id] = res
            logger.debug(f"Results stored for batch {batch_idx + 1}")

        logger.info(f"Batch prediction completed, total results: {len(all_results)}")
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

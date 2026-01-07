"""Segmenting remote sensing images with the Segment Anything Model 3 (SAM3).
https://github.com/facebookresearch/sam3
"""

import glob
import os
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image
import rasterio
from tqdm import tqdm
from rasterio.windows import Window


SAM3_META_AVAILABLE = False

try:
    from transformers import (
        Sam3Model,
        Sam3Processor as TransformersSam3Processor,
    )  # pyright: ignore[reportAttributeAccessIssue, reportMissingImports]
    import torch

    SAM3_TRANSFORMERS_AVAILABLE = True
except ImportError:
    SAM3_TRANSFORMERS_AVAILABLE = False

try:
    from skimage.color import lab2rgb, rgb2lab
    from sklearn.cluster import KMeans
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgb
    import matplotlib.patches as patches
except ImportError as e:
    print(f"To use SamGeo 3, install it as:\n\tpip install segment-geospatial[samgeo3]")

from samgeo import common

try:
    from transformers.utils import logging as hf_logging

    hf_logging.set_verbosity_error()  # silence HF load reports
except ImportError:
    pass


class SamGeo3:
    """The main class for segmenting geospatial data with the Segment Anything Model 3 (SAM3)."""

    def __init__(
        self,
        model_id="data/sam3",
        device=None,
        checkpoint_path=None,
        confidence_threshold=0.5,
        mask_threshold=0.5,
    ) -> None:
        """
        Initializes the SamGeo3 class.

        Args:
            model_id (str): Model ID for Transformers backend (e.g., 'facebook/sam3').
                Only used when backend='transformers'.
            bpe_path (str, optional): Path to the BPE tokenizer vocabulary (Meta backend only).
            device (str, optional): Device to load the model on ('cuda' or 'cpu').
            eval_mode (bool, optional): Whether to set the model to evaluation mode (Meta backend only).
            checkpoint_path (str, optional): Optional path to model checkpoint (Meta backend only).
            load_from_HF (bool, optional): Whether to load the model from HuggingFace (Meta backend only).
            enable_segmentation (bool, optional): Whether to enable segmentation head (Meta backend only).
            enable_inst_interactivity (bool, optional): Whether to enable instance interactivity
                (SAM 1 task) (Meta backend only). Set to True to use predict_inst() and
                predict_inst_batch() methods for interactive point and box prompts.
                When True, the model loads additional components for SAM1-style
                interactive instance segmentation. Defaults to False.
            compile_mode (bool, optional): To enable compilation, set to "default" (Meta backend only).
            confidence_threshold (float, optional): Confidence threshold for the model.
            mask_threshold (float, optional): Mask threshold for post-processing (Transformers backend only).
            **kwargs: Additional keyword arguments.

        Example:
            >>> # For text-based segmentation
            >>> sam = SamGeo3(backend="meta")
            >>> sam.set_image("image.jpg")
            >>> sam.generate_masks("tree")
            >>>
            >>> # For interactive point/box prompts (SAM1-style)
            >>> sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
            >>> sam.set_image("image.jpg")
            >>> masks, scores, logits = sam.predict_inst(
            ...     point_coords=np.array([[520, 375]]),
            ...     point_labels=np.array([1]),
            ... )
        """
        if not SAM3_TRANSFORMERS_AVAILABLE:
            raise ImportError(
                "Transformers SAM3 is not available. Please install it as:\n\tpip install transformers torch"
            )

        if device is None:
            device = common.get_device()

        self.device = device
        self.confidence_threshold = confidence_threshold
        self.mask_threshold = mask_threshold
        self.model_id = model_id
        self.model_version = "sam3"

        self._init_transformers_backend(
            model_id=model_id,
            device=device,
        )

        # Common attributes
        self.predictor = None
        self.masks = None
        self.boxes = None
        self.scores = None
        self.logits = None
        self.objects = None
        self.prediction = None
        self.source = None
        self.image = None
        self.image_height = None
        self.image_width = None
        self.inference_state = None

    def _init_transformers_backend(self, model_id, device):
        """Initialize Transformers SAM3 backend."""
        self.model = Sam3Model.from_pretrained(model_id).to(device)
        self.processor = TransformersSam3Processor.from_pretrained(model_id)

    def set_confidence_threshold(self, threshold: float, state=None):
        """Sets the confidence threshold for the masks.
        Args:
            threshold (float): The confidence threshold.
            state (optional): An optional state object to pass to the processor's set_confidence_threshold method (Meta backend only).
        """
        # For transformers backend, the threshold is stored and used during generate_masks
        self.confidence_threshold = threshold

    def _set_image_with_string(
        self, image_path: str, bands: Optional[List[int]] = None
    ):
        if image_path.startswith("http"):
            image = common.download_file(image_path)

        if not os.path.exists(image_path):
            raise ValueError(f"Input path {image_path} does not exist.")

        self.source = image_path

        # Check if image is a GeoTIFF and handle band selection
        if image_path.lower().endswith((".tif", ".tiff")):

            with rasterio.open(image_path) as src:
                if bands is not None:
                    # Validate band indices (1-based)
                    array = self.__read_image_with_bands(bands, src)
                else:
                    array = self.__read_image_without_bands(src)
                # Transpose from (bands, height, width) to (height, width, bands)
                image = self.__format_array_as_image(array)
        else:
            image = self._load_non_tiff_path(image_path)
        self.image = image

    @staticmethod
    def __format_array_as_image(array: np.ndarray) -> np.ndarray:
        array = np.transpose(array, (1, 2, 0))

        # Normalize to 8-bit (0-255) range
        array = array.astype(np.float32)
        array -= array.min()
        if array.max() > 0:
            array /= array.max()
        array *= 255
        image = array.astype(np.uint8)
        return image

    @staticmethod
    def __read_image_without_bands(src) -> np.ndarray:
        # Read all bands
        array = src.read()
        # If more than 3 bands, use first 3
        if array.shape[0] >= 3:
            array = array[:3, :, :]
        elif array.shape[0] == 1:
            array = np.repeat(array, 3, axis=0)
        elif array.shape[0] == 2:
            # Repeat the first band to make 3 bands: [band1, band2, band1]
            array = np.concatenate([array, array[0:1, :, :]], axis=0)
        return array

    @staticmethod
    def __read_image_with_bands(bands, src) -> np.ndarray:
        if len(bands) != 3:
            raise ValueError("bands must contain exactly 3 band indices for RGB.")
        for band in bands:
            if band < 1 or band > src.count:
                raise ValueError(
                    f"Band index {band} is out of range. "
                    f"Image has {src.count} bands (1-indexed)."
                )
                # Read specified bands (rasterio uses 1-based indexing)
        array = np.stack([src.read(b) for b in bands], axis=0)
        return array

    def _load_non_tiff_path(self, image_path):
        image = cv2.imread(image_path)
        if image is not None:
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def set_image(
        self,
        image_input: Union[str, np.ndarray, Image.Image],
        bands: Optional[List[int]] = None,
    ) -> None:
        """Set the input image as a numpy array.

        Args:
            image (Union[str, np.ndarray, Image]): The input image as a path,
                a numpy array, or an Image.
            bands (List[int], optional): List of band indices (1-based) to use for RGB
                when the input is a GeoTIFF with more than 3 bands. For example,
                [4, 3, 2] for NIR-R-G false color composite. If None, uses the
                first 3 bands for multi-band images. Defaults to None.
        """
        if isinstance(image_input, str):
            self._set_image_with_string(image_input, bands=bands)

        elif isinstance(image_input, np.ndarray):
            self.image = image_input
            self.source = None
        elif isinstance(image_input, Image.Image):
            self.image = np.array(image_input)
            self.source = None
        else:
            raise ValueError(
                "Input image must be either a path, numpy array, or PIL Image."
            )
        if self.image is None:
            raise ValueError("Failed to load image.")
        self.image_height, self.image_width = self.image.shape[:2]

        # Convert to PIL Image for processing
        image_for_processor = Image.fromarray(self.image)
        self.pil_image = image_for_processor

    def generate_masks(
        self,
        prompt: str,
        min_size: int = 0,
        max_size: Optional[int] = None,
        quiet: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Generate masks for the input image using SAM3.

        Args:
            prompt (str): The text prompt describing the objects to segment.
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).
            quiet (bool): If True, suppress progress messages. Defaults to False.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries containing the generated masks.
        """
        if not hasattr(self, "pil_image"):
            raise ValueError("No image set. Please call set_image() first.")

        # Prepare inputs
        inputs = self.processor(
            images=self.pil_image, text=prompt, return_tensors="pt"
        ).to(self.device)

        # Get original sizes for post-processing
        original_sizes = self._get_original_sizes(inputs)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        self._post_process_model_results(original_sizes, outputs)
        self._convert_results_to_numpy()

        # Filter masks by size if min_size or max_size is specified
        if min_size > 0 or max_size is not None:
            self._filter_masks_by_size(min_size, max_size)

        if not quiet:
            num_objects = len(self.masks)
            self._print_num_objects(num_objects)

    def _post_process_model_results(self, original_sizes, outputs):
        results = self.processor.post_process_instance_segmentation(
            outputs,
            threshold=self.confidence_threshold,
            mask_threshold=self.mask_threshold,
            target_sizes=original_sizes,
        )[0]

        # Convert results to match Meta backend format
        self.masks = results["masks"]
        self.boxes = results["boxes"]
        self.scores = results["scores"]

    @staticmethod
    def _print_num_objects(num_objects):
        if num_objects == 0:
            print("No objects found. Please try a different prompt.")
        elif num_objects == 1:
            print("Found one object.")
        else:
            print(f"Found {num_objects} objects.")

    def _get_original_sizes(self, inputs):
        original_sizes = inputs.get("original_sizes")
        if original_sizes is not None:
            original_sizes = original_sizes.tolist()
        else:
            original_sizes = [[self.image_height, self.image_width]]
        return original_sizes

    def generate_masks_tiled(
        self,
        source: str,
        prompt: str,
        output: str,
        tile_size: int = 1024,
        overlap: int = 128,
        min_size: int = 0,
        max_size: Optional[int] = None,
        unique: bool = True,
        dtype: str = "uint32",
        bands: Optional[List[int]] = None,
        batch_size: int = 1,
        verbose: bool = True,
    ) -> str:
        """
        Generate masks for large GeoTIFF images using a sliding window approach.

        This method processes large images tile by tile to avoid GPU memory issues.
        The tiles are processed with overlap to ensure seamless mask merging at
        boundaries. Each detected object gets a unique ID that is consistent
        across the entire image.

        Args:
            source (str): Path to the input GeoTIFF image.
            prompt (str): The text prompt describing the objects to segment.
            output (str): Path to the output GeoTIFF file.
            tile_size (int): Size of each tile in pixels. Defaults to 1024.
            overlap (int): Overlap between adjacent tiles in pixels. Defaults to 128.
                Higher overlap helps with better boundary merging but increases
                processing time.
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).
            unique (bool): If True, each mask gets a unique value. If False, binary
                mask (0 or 1). Defaults to True.
            dtype (str): Data type for the output array. Use 'uint32' for large
                numbers of objects, 'uint16' for up to 65535 objects, or 'uint8'
                for up to 255 objects. Defaults to 'uint32'.
            bands (List[int], optional): List of band indices (1-based) to use for RGB
                when the input has more than 3 bands. If None, uses first 3 bands.
            batch_size (int): Number of tiles to process at once (future use).
                Defaults to 1.
            verbose (bool): Whether to print progress information. Defaults to True.
            **kwargs: Additional keyword arguments.

        Returns:
            str: Path to the output GeoTIFF file.

        Example:
            >>> sam = SamGeo3(backend="meta")
            >>> sam.generate_masks_tiled(
            ...     source="large_satellite_image.tif",
            ...     prompt="building",
            ...     output="buildings_mask.tif",
            ...     tile_size=1024,
            ...     overlap=128,
            ... )
        """

        if not source.lower().endswith((".tif", ".tiff")):
            raise ValueError("Source must be a GeoTIFF file for tiled processing.")

        if not os.path.exists(source):
            raise ValueError(f"Source file not found: {source}")

        if tile_size <= overlap:
            raise ValueError("tile_size must be greater than overlap")

        # Open the source file to get metadata
        with rasterio.open(source) as src:
            img_height = src.height
            img_width = src.width
            profile = src.profile.copy()

        if verbose:
            print(f"Processing image: {img_width} x {img_height} pixels")
            print(f"Tile size: {tile_size}, Overlap: {overlap}")

        # Calculate the number of tiles
        step = tile_size - overlap
        n_tiles_x = max(1, (img_width - overlap + step - 1) // step)
        n_tiles_y = max(1, (img_height - overlap + step - 1) // step)
        total_tiles = n_tiles_x * n_tiles_y

        if verbose:
            print(f"Total tiles to process: {total_tiles} ({n_tiles_x} x {n_tiles_y})")

        # Determine output dtype
        if dtype == "uint8":
            np_dtype = np.uint8
            max_objects = 255
        elif dtype == "uint16":
            np_dtype = np.uint16
            max_objects = 65535
        elif dtype == "uint32":
            np_dtype = np.uint32
            max_objects = 4294967295
        else:
            np_dtype = np.uint32
            max_objects = 4294967295

        # Create output array in memory (for smaller images) or use memory-mapped file
        # For very large images, you might want to use rasterio windowed writing
        output_mask = np.zeros((img_height, img_width), dtype=np_dtype)

        # Track unique object IDs across all tiles
        current_max_id = 0
        total_objects = 0

        # Process each tile
        tile_iterator = tqdm(
            range(total_tiles),
            desc="Processing tiles",
            disable=not verbose,
        )

        for tile_idx in tile_iterator:
            # Calculate tile position
            tile_y = tile_idx // n_tiles_x
            tile_x = tile_idx % n_tiles_x

            # Calculate window coordinates
            x_start = tile_x * step
            y_start = tile_y * step

            # Ensure we don't go beyond image bounds
            x_end = min(x_start + tile_size, img_width)
            y_end = min(y_start + tile_size, img_height)

            # Adjust start if we're at the edge
            if x_end - x_start < tile_size and x_start > 0:
                x_start = max(0, x_end - tile_size)
            if y_end - y_start < tile_size and y_start > 0:
                y_start = max(0, y_end - tile_size)

            window_width = x_end - x_start
            window_height = y_end - y_start

            # Read tile from source
            with rasterio.open(source) as src:
                window = Window(x_start, y_start, window_width, window_height)
                if bands is not None:
                    tile_data = np.stack(
                        [src.read(b, window=window) for b in bands], axis=0
                    )
                else:
                    tile_data = src.read(window=window)
                    if tile_data.shape[0] >= 3:
                        tile_data = tile_data[:3, :, :]
                    elif tile_data.shape[0] == 1:
                        tile_data = np.repeat(tile_data, 3, axis=0)
                    elif tile_data.shape[0] == 2:
                        tile_data = np.concatenate(
                            [tile_data, tile_data[0:1, :, :]], axis=0
                        )

            # Transpose to (height, width, channels)
            tile_data = np.transpose(tile_data, (1, 2, 0))

            # Normalize to 8-bit
            tile_data = tile_data.astype(np.float32)
            tile_data -= tile_data.min()
            if tile_data.max() > 0:
                tile_data /= tile_data.max()
            tile_data *= 255
            tile_image = tile_data.astype(np.uint8)

            # Process the tile
            try:
                # Set image for the tile
                self.image = tile_image
                self.image_height, self.image_width = tile_image.shape[:2]
                self.source = None  # Don't need georef for individual tiles

                # Initialize inference state for this tile
                pil_image = Image.fromarray(tile_image)
                self.pil_image = pil_image

                if self.backend == "meta":
                    self.inference_state = self.processor.set_image(pil_image)
                else:
                    # For transformers backend, process directly
                    pass

                # Generate masks for this tile (quiet=True to avoid per-tile messages)
                self.generate_masks(
                    prompt, min_size=min_size, max_size=max_size, quiet=True
                )

                # Get masks for this tile
                tile_masks = self.masks

                if tile_masks is not None and len(tile_masks) > 0:
                    # Create a mask array for this tile
                    tile_mask_array = np.zeros(
                        (window_height, window_width), dtype=np_dtype
                    )

                    for mask in tile_masks:
                        # Convert mask to numpy
                        if hasattr(mask, "cpu"):
                            mask_np = mask.squeeze().cpu().numpy()
                        elif hasattr(mask, "numpy"):
                            mask_np = mask.squeeze().numpy()
                        else:
                            mask_np = (
                                mask.squeeze() if hasattr(mask, "squeeze") else mask
                            )

                        if mask_np.ndim > 2:
                            mask_np = mask_np[0]

                        # Resize mask to tile size if needed
                        if mask_np.shape != (window_height, window_width):
                            mask_np = cv2.resize(
                                mask_np.astype(np.float32),
                                (window_width, window_height),
                                interpolation=cv2.INTER_NEAREST,
                            )

                        mask_bool = mask_np > 0
                        mask_size = np.sum(mask_bool)

                        # Filter by size
                        if mask_size < min_size:
                            continue
                        if max_size is not None and mask_size > max_size:
                            continue

                        if unique:
                            current_max_id += 1
                            if current_max_id > max_objects:
                                raise ValueError(
                                    f"Maximum number of objects ({max_objects}) exceeded. "
                                    "Consider using a larger dtype or reducing the number of objects."
                                )
                            tile_mask_array[mask_bool] = current_max_id
                        else:
                            tile_mask_array[mask_bool] = 1

                        total_objects += 1

                    # Merge tile mask into output mask
                    # For overlapping regions, use the tile's values if they are non-zero
                    # This simple approach works well for most cases
                    self._merge_tile_mask(
                        output_mask,
                        tile_mask_array,
                        x_start,
                        y_start,
                        x_end,
                        y_end,
                        overlap,
                        tile_x,
                        tile_y,
                        n_tiles_x,
                        n_tiles_y,
                    )

            except Exception as e:
                if verbose:
                    print(f"Warning: Failed to process tile ({tile_x}, {tile_y}): {e}")
                continue

            # Clear GPU memory
            self.masks = None
            self.boxes = None
            self.scores = None
            if hasattr(self, "inference_state"):
                self.inference_state = None
            # Additionally clear PyTorch CUDA cache, if available, to free GPU memory
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except ImportError:
                # If torch is not installed, skip CUDA cache clearing
                pass
        # Update output profile
        profile.update(
            {
                "count": 1,
                "dtype": dtype,
                "compress": "deflate",
            }
        )

        # Save the output
        with rasterio.open(output, "w", **profile) as dst:
            dst.write(output_mask, 1)

        if verbose:
            print(f"Saved mask to {output}")
            print(f"Total objects found: {total_objects}")

        # Store result for potential visualization
        self.objects = output_mask
        self.source = source

        return output

    @staticmethod
    def _merge_tile_mask(
        output_mask: np.ndarray,
        tile_mask: np.ndarray,
        x_start: int,
        y_start: int,
        x_end: int,
        y_end: int,
        overlap: int,
        tile_x: int,
        tile_y: int,
        n_tiles_x: int,
        n_tiles_y: int,
    ) -> None:
        """
        Merge a tile mask into the output mask, handling overlapping regions.

        For overlapping regions, this uses a blending approach where we prioritize
        the current tile's mask in the non-overlapping core region, and for the
        overlap region, we keep existing values unless they are zero.

        Args:
            output_mask: The full output mask array.
            tile_mask: The mask from the current tile.
            x_start, y_start: Start coordinates of the tile in the output.
            x_end, y_end: End coordinates of the tile in the output.
            overlap: The overlap size.
            tile_x, tile_y: Tile indices.
            n_tiles_x, n_tiles_y: Total number of tiles in each direction.
        """
        tile_height = y_end - y_start
        tile_width = x_end - x_start

        # Calculate the core region (non-overlapping part)
        # The overlap should be split between adjacent tiles
        left_overlap = overlap // 2 if tile_x > 0 else 0
        right_overlap = overlap // 2 if tile_x < n_tiles_x - 1 else 0
        top_overlap = overlap // 2 if tile_y > 0 else 0
        bottom_overlap = overlap // 2 if tile_y < n_tiles_y - 1 else 0

        # Core region in tile coordinates
        core_x_start = left_overlap
        core_x_end = tile_width - right_overlap
        core_y_start = top_overlap
        core_y_end = tile_height - bottom_overlap

        # Copy core region (always overwrite)
        out_y_start = y_start + core_y_start
        out_y_end = y_start + core_y_end
        out_x_start = x_start + core_x_start
        out_x_end = x_start + core_x_end

        output_mask[out_y_start:out_y_end, out_x_start:out_x_end] = tile_mask[
            core_y_start:core_y_end, core_x_start:core_x_end
        ]

        # Handle overlap regions - only update if output is zero
        # Top overlap
        if top_overlap > 0:
            region = output_mask[y_start : y_start + top_overlap, out_x_start:out_x_end]
            tile_region = tile_mask[0:top_overlap, core_x_start:core_x_end]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Bottom overlap
        if bottom_overlap > 0:
            region = output_mask[out_y_end:y_end, out_x_start:out_x_end]
            tile_region = tile_mask[core_y_end:tile_height, core_x_start:core_x_end]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Left overlap
        if left_overlap > 0:
            region = output_mask[
                out_y_start:out_y_end, x_start : x_start + left_overlap
            ]
            tile_region = tile_mask[core_y_start:core_y_end, 0:left_overlap]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Right overlap
        if right_overlap > 0:
            region = output_mask[out_y_start:out_y_end, out_x_end:x_end]
            tile_region = tile_mask[core_y_start:core_y_end, core_x_end:tile_width]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Corner overlaps
        # Top-left
        if top_overlap > 0 and left_overlap > 0:
            region = output_mask[
                y_start : y_start + top_overlap, x_start : x_start + left_overlap
            ]
            tile_region = tile_mask[0:top_overlap, 0:left_overlap]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Top-right
        if top_overlap > 0 and right_overlap > 0:
            region = output_mask[y_start : y_start + top_overlap, out_x_end:x_end]
            tile_region = tile_mask[0:top_overlap, core_x_end:tile_width]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Bottom-left
        if bottom_overlap > 0 and left_overlap > 0:
            region = output_mask[out_y_end:y_end, x_start : x_start + left_overlap]
            tile_region = tile_mask[core_y_end:tile_height, 0:left_overlap]
            mask = region == 0
            region[mask] = tile_region[mask]

        # Bottom-right
        if bottom_overlap > 0 and right_overlap > 0:
            region = output_mask[out_y_end:y_end, out_x_end:x_end]
            tile_region = tile_mask[core_y_end:tile_height, core_x_end:tile_width]
            mask = region == 0
            region[mask] = tile_region[mask]

    def _transform_boxes_to_pixel_coords(
        self,
        boxes: List[List[float]],
        box_crs: str,
    ) -> List[List[float]]:
        """Transform boxes from given CRS to pixel coordinates.

        Args:
            boxes (List[List[float]]): List of bounding boxes in XYXY format
                [[xmin, ymin, xmax, ymax], ...].
            box_crs (str): Coordinate reference system for box coordinates
                (e.g., "EPSG:4326" for lat/lon).
        Returns:
            List[List[float]]: List of bounding boxes in pixel coordinates.
        """
        if self.source is None:
            raise ValueError(
                "Source image is not set. Cannot transform boxes without georeference."
            )

        pixel_boxes = []
        for box in boxes:
            xmin, ymin, xmax, ymax = box

            # Transform min corner
            min_coords = np.array([[xmin, ymin]])
            min_xy, _ = common.coords_to_xy(
                self.source, min_coords, box_crs, return_out_of_bounds=True
            )

            # Transform max corner
            max_coords = np.array([[xmax, ymax]])
            max_xy, _ = common.coords_to_xy(
                self.source, max_coords, box_crs, return_out_of_bounds=True
            )

            # Convert to pixel coordinates and ensure correct min/max order
            # (geographic y increases north, pixel y increases down)
            x1_px = min_xy[0][0]
            y1_px = min_xy[0][1]
            x2_px = max_xy[0][0]
            y2_px = max_xy[0][1]

            # Ensure we have correct min/max values
            x_min_px = min(x1_px, x2_px)
            y_min_px = min(y1_px, y2_px)
            x_max_px = max(x1_px, x2_px)
            y_max_px = max(y1_px, y2_px)

            pixel_boxes.append([x_min_px, y_min_px, x_max_px, y_max_px])

        return pixel_boxes

    def generate_masks_by_boxes(
        self,
        boxes: List[List[float]],
        box_labels: Optional[List[bool]] = None,
        box_crs: Optional[str] = None,
        min_size: int = 0,
        max_size: Optional[int] = None,
        quiet: bool = False,
    ) -> Dict[str, Any]:
        """
        Generate masks using bounding box prompts.

        Args:
            boxes (List[List[float]]): List of bounding boxes in XYXY format
                [[xmin, ymin, xmax, ymax], ...].
                If box_crs is None: pixel coordinates.
                If box_crs is specified: coordinates in the given CRS (e.g., "EPSG:4326").
            box_labels (List[bool], optional): List of boolean labels for each box.
                True for positive prompt (include), False for negative prompt (exclude).
                If None, all boxes are treated as positive prompts.
            box_crs (str, optional): Coordinate reference system for box coordinates
                (e.g., "EPSG:4326" for lat/lon). Only used if the source image is a GeoTIFF.
                If None, boxes are assumed to be in pixel coordinates.
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out. Defaults to 0.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out. Defaults to None (no maximum).
            **kwargs: Additional keyword arguments.

        Returns:
            Dict[str, Any]: Dictionary containing masks, boxes, and scores.

        Example:
            # For pixel coordinates:
            boxes = [[100, 200, 300, 400]]
            sam.generate_masks_by_boxes(boxes)

            # For geographic coordinates (GeoTIFF):
            boxes = [[-122.5, 37.7, -122.4, 37.8]]  # [lon_min, lat_min, lon_max, lat_max]
            sam.generate_masks_by_boxes(boxes, box_crs="EPSG:4326")
        """
        if not hasattr(self, "pil_image"):
            raise ValueError("No image set. Please call set_image() first.")
        if box_labels is None:
            box_labels = [True] * len(boxes)
        if len(boxes) != len(box_labels):
            raise ValueError(
                f"Number of boxes ({len(boxes)}) must match number of labels ({len(box_labels)})"
            )
        # Transform boxes from CRS to pixel coordinates if needed
        if box_crs is not None and self.source is not None:
            boxes = self._transform_boxes_to_pixel_coords(boxes, box_crs)

        # For Transformers backend, process boxes with the processor
        # Convert boxes to the format expected by Transformers
        # Transformers expects boxes in XYXY format with 3 levels of nesting:
        # [image level, box level, box coordinates]
        # Also convert numpy types to Python native types
        input_boxes = [
            [[float(coord) for coord in box] for box in boxes]
        ]  # Wrap in list for image level and convert to float

        # Prepare inputs with boxes
        inputs = self.processor(
            images=self.pil_image, input_boxes=input_boxes, return_tensors="pt"
        ).to(self.device)

        # Get original sizes for post-processing
        original_sizes = self._get_original_sizes(inputs)

        # Run inference
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Post-process results
        self._post_process_model_results(original_sizes, outputs)

        # Convert tensors to numpy to free GPU memory
        self._convert_results_to_numpy()

        # Filter masks by size if min_size or max_size is specified
        if min_size > 0 or max_size is not None:
            self._filter_masks_by_size(min_size, max_size)
        if not quiet:
            num_objects = len(self.masks)
            self._print_num_objects(num_objects)

    def _convert_results_to_numpy(self) -> None:
        """Convert masks, boxes, and scores from tensors to numpy arrays.

        This frees GPU memory by moving data to CPU and converting to numpy.
        """
        if self.masks is None:
            return

        # Convert masks to numpy
        converted_masks = []
        for mask in self.masks:
            if hasattr(mask, "cpu"):
                # PyTorch tensor on GPU
                mask_np = mask.cpu().numpy()
            elif hasattr(mask, "numpy"):
                # PyTorch tensor on CPU
                mask_np = mask.numpy()
            else:
                # Already numpy or other array-like
                mask_np = np.asarray(mask)
            converted_masks.append(mask_np)
        self.masks = converted_masks

        # Convert boxes to numpy
        if self.boxes is not None:
            converted_boxes = []
            for box in self.boxes:
                if hasattr(box, "cpu"):
                    box_np = box.cpu().numpy()
                elif hasattr(box, "numpy"):
                    box_np = box.numpy()
                else:
                    box_np = np.asarray(box)
                converted_boxes.append(box_np)
            self.boxes = converted_boxes

        # Convert scores to numpy/float
        if self.scores is not None:
            converted_scores = []
            for score in self.scores:
                if hasattr(score, "cpu"):
                    score_val = (
                        score.cpu().item()
                        if score.numel() == 1
                        else score.cpu().numpy()
                    )
                elif hasattr(score, "item"):
                    score_val = score.item()
                elif hasattr(score, "numpy"):
                    score_val = score.numpy()
                else:
                    score_val = float(score)
                converted_scores.append(score_val)
            self.scores = converted_scores

    def _filter_masks_by_size(
        self, min_size: int = 0, max_size: Optional[int] = None
    ) -> None:
        """Filter masks by size.

        Args:
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out.
        """
        if self.masks is None or len(self.masks) == 0:
            return

        filtered_masks = []
        filtered_boxes = []
        filtered_scores = []

        for i, mask in enumerate(self.masks):
            # Convert mask to numpy array if it's a tensor
            if hasattr(mask, "cpu"):
                mask_np = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = mask.squeeze() if hasattr(mask, "squeeze") else mask

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            # Convert to boolean and calculate mask size
            mask_bool = mask_np > 0
            mask_size = np.sum(mask_bool)

            # Filter by size
            if mask_size < min_size:
                continue
            if max_size is not None and mask_size > max_size:
                continue

            # Keep this mask
            filtered_masks.append(self.masks[i])
            if self.boxes is not None and len(self.boxes) > i:
                filtered_boxes.append(self.boxes[i])
            if self.scores is not None and len(self.scores) > i:
                filtered_scores.append(self.scores[i])

        # Update the stored masks, boxes, and scores
        self.masks = filtered_masks
        self.boxes = filtered_boxes if filtered_boxes else self.boxes
        self.scores = filtered_scores if filtered_scores else self.scores

    def show_boxes(
        self,
        boxes: List[List[float]],
        box_labels: Optional[List[bool]] = None,
        box_crs: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        positive_color: Tuple[int, int, int] = (0, 255, 0),
        negative_color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 3,
    ) -> None:
        """
        Visualize bounding boxes on the image.

        Args:
            boxes (List[List[float]]): List of bounding boxes in XYXY format
                [[xmin, ymin, xmax, ymax], ...].
                If box_crs is None: pixel coordinates.
                If box_crs is specified: coordinates in the given CRS.
            box_labels (List[bool], optional): List of boolean labels for each box.
                True (positive) shown in green, False (negative) shown in red.
                If None, all boxes shown in green.
            box_crs (str, optional): Coordinate reference system for box coordinates
                (e.g., "EPSG:4326"). If None, boxes are in pixel coordinates.
            figsize (Tuple[int, int]): Figure size for display.
            axis (str): Whether to show axis ("on" or "off").
            positive_color (Tuple[int, int, int]): RGB color for positive boxes.
            negative_color (Tuple[int, int, int]): RGB color for negative boxes.
            thickness (int): Line thickness for box borders.
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")

        if box_labels is None:
            box_labels = [True] * len(boxes)

        # Transform boxes from CRS to pixel coordinates if needed
        if box_crs is not None and self.source is not None:
            pixel_boxes = []
            for box in boxes:
                xmin, ymin, xmax, ymax = box

                # Transform min corner
                min_coords = np.array([[xmin, ymin]])
                min_xy, _ = common.coords_to_xy(
                    self.source, min_coords, box_crs, return_out_of_bounds=True
                )

                # Transform max corner
                max_coords = np.array([[xmax, ymax]])
                max_xy, _ = common.coords_to_xy(
                    self.source, max_coords, box_crs, return_out_of_bounds=True
                )

                # Convert to pixel coordinates and ensure correct min/max order
                # (geographic y increases north, pixel y increases down)
                x1_px = min_xy[0][0]
                y1_px = min_xy[0][1]
                x2_px = max_xy[0][0]
                y2_px = max_xy[0][1]

                # Ensure we have correct min/max values
                x_min_px = min(x1_px, x2_px)
                y_min_px = min(y1_px, y2_px)
                x_max_px = max(x1_px, x2_px)
                y_max_px = max(y1_px, y2_px)

                pixel_boxes.append([x_min_px, y_min_px, x_max_px, y_max_px])

            boxes = pixel_boxes

        # Convert image to PIL if needed
        if isinstance(self.image, np.ndarray):
            img = Image.fromarray(self.image)
        else:
            img = self.image

        # Draw each box
        for box, label in zip(boxes, box_labels):
            # Convert XYXY to XYWH for drawing
            xmin, ymin, xmax, ymax = box
            box_xywh = [xmin, ymin, xmax - xmin, ymax - ymin]

            # Choose color based on label
            color = positive_color if label else negative_color

            # Draw box
            img = draw_box_on_image(img, box_xywh, color=color, thickness=thickness)

        # Display
        plt.figure(figsize=figsize)
        plt.imshow(img)
        plt.axis(axis)
        plt.show()

    def show_points(
        self,
        point_coords: List[List[float]],
        point_labels: Optional[List[int]] = None,
        point_crs: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        foreground_color: str = "green",
        background_color: str = "red",
        marker: str = "*",
        marker_size: int = 375,
    ) -> None:
        """
        Visualize point prompts on the image.

        Args:
            point_coords (List[List[float]]): List of point coordinates [[x, y], ...].
                If point_crs is None: pixel coordinates.
                If point_crs is specified: coordinates in the given CRS.
            point_labels (List[int], optional): List of labels for each point.
                1 = foreground (shown in green), 0 = background (shown in red).
                If None, all points shown as foreground.
            point_crs (str, optional): Coordinate reference system for point coordinates
                (e.g., "EPSG:4326"). If None, points are in pixel coordinates.
            figsize (Tuple[int, int]): Figure size for display.
            axis (str): Whether to show axis ("on" or "off").
            foreground_color (str): Color for foreground points (label=1).
            background_color (str): Color for background points (label=0).
            marker (str): Marker style for points.
            marker_size (int): Size of the markers.

        Example:
            sam.show_points([[520, 375]], [1])  # Single foreground point
            sam.show_points([[500, 375], [600, 400]], [1, 0])  # Mixed points
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")

        if point_labels is None:
            point_labels = [1] * len(point_coords)

        # Convert to numpy arrays
        point_coords = np.array(point_coords)
        point_labels = np.array(point_labels)

        # Transform points from CRS to pixel coordinates if needed
        if point_crs is not None and self.source is not None:
            point_coords, _ = common.coords_to_xy(
                self.source, point_coords, point_crs, return_out_of_bounds=True
            )

        # Display image
        plt.figure(figsize=figsize)
        plt.imshow(self.image)

        # Plot foreground points
        fg_mask = point_labels == 1
        if np.any(fg_mask):
            plt.scatter(
                point_coords[fg_mask, 0],
                point_coords[fg_mask, 1],
                color=foreground_color,
                marker=marker,
                s=marker_size,
                edgecolor="white",
                linewidth=1.25,
            )

        # Plot background points
        bg_mask = point_labels == 0
        if np.any(bg_mask):
            plt.scatter(
                point_coords[bg_mask, 0],
                point_coords[bg_mask, 1],
                color=background_color,
                marker=marker,
                s=marker_size,
                edgecolor="white",
                linewidth=1.25,
            )

        plt.axis(axis)
        plt.show()

    def save_masks(
        self,
        output: Optional[str] = None,
        unique: bool = True,
        min_size: int = 0,
        max_size: Optional[int] = None,
        dtype: str = "uint8",
        save_scores: Optional[str] = None,
        **kwargs: Any,
    ) -> None:
        """Save the generated masks to a file or generate mask array for visualization.

        If the input image is a GeoTIFF, the output will be saved as a GeoTIFF
        with the same georeferencing information. Otherwise, it will be saved as PNG.

        Args:
            output (str, optional): The path to the output file. If None, only generates
                the mask array in memory (self.objects) without saving to disk.
            unique (bool): If True, each mask gets a unique value (1, 2, 3, ...).
                If False, all masks are combined into a binary mask (0 or 255).
            min_size (int): Minimum mask size in pixels. Masks smaller than this
                will be filtered out.
            max_size (int, optional): Maximum mask size in pixels. Masks larger than
                this will be filtered out.
            dtype (str): Data type for the output array.
            save_scores (str, optional): If provided, saves a confidence score map
                to this path. Each pixel will have the confidence score of its mask.
                The output format (GeoTIFF or PNG) follows the same logic as the mask output.
            **kwargs: Additional keyword arguments passed to common.array_to_image().
        """
        if self.masks is None or len(self.masks) == 0:
            raise ValueError("No masks found. Please run generate_masks() first.")

        if save_scores is not None and self.scores is None:
            raise ValueError("No scores found. Cannot save scores.")

        # Create empty array for combined masks
        mask_array = np.zeros(
            (self.image_height, self.image_width),
            dtype=np.uint32 if unique else np.uint8,
        )

        # Create empty array for scores if requested
        if save_scores is not None:
            scores_array = np.zeros(
                (self.image_height, self.image_width), dtype=np.float32
            )

        # Process each mask
        valid_mask_count = 0
        mask_index = 0
        for mask in self.masks:
            # Convert mask to numpy array if it's a tensor
            if hasattr(mask, "cpu"):
                mask_np = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = mask.squeeze() if hasattr(mask, "squeeze") else mask

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            # Convert to boolean
            mask_bool = mask_np > 0

            # Calculate mask size
            mask_size = np.sum(mask_bool)

            # Filter by size
            if mask_size < min_size:
                mask_index += 1
                continue
            if max_size is not None and mask_size > max_size:
                mask_index += 1
                continue

            # Get confidence score for this mask
            if save_scores is not None:
                if hasattr(self.scores[mask_index], "item"):
                    score = self.scores[mask_index].item()
                else:
                    score = float(self.scores[mask_index])

            # Add mask to array
            if unique:
                # Assign unique value to each mask (starting from 1)
                mask_value = valid_mask_count + 1
                mask_array[mask_bool] = mask_value
            else:
                # Binary mask: all foreground pixels are 255
                mask_array[mask_bool] = 255

            # Add score to scores array
            if save_scores is not None:
                scores_array[mask_bool] = score

            valid_mask_count += 1
            mask_index += 1

        if valid_mask_count == 0:
            print("No masks met the size criteria.")
            return

        # Convert to requested dtype
        if dtype == "uint8":
            if unique and valid_mask_count > 255:
                print(
                    f"Warning: {valid_mask_count} masks found, but uint8 can only represent 255 unique values. Consider using dtype='uint16'."
                )
            mask_array = mask_array.astype(np.uint8)
        elif dtype == "uint16":
            mask_array = mask_array.astype(np.uint16)
        elif dtype == "int32":
            mask_array = mask_array.astype(np.int32)
        else:
            mask_array = mask_array.astype(dtype)

        # Store the mask array for visualization
        self.objects = mask_array

        # Only save to file if output path is provided
        if output is not None:
            # Save using common utility which handles GeoTIFF georeferencing
            common.array_to_image(
                mask_array, output, self.source, dtype=dtype, **kwargs
            )
            print(f"Saved {valid_mask_count} mask(s) to {output}")

            # Save scores if requested
            if save_scores is not None:
                common.array_to_image(
                    scores_array, save_scores, self.source, dtype="float32", **kwargs
                )
                print(f"Saved confidence scores to {save_scores}")

    def save_prediction(
        self,
        output: str,
        index: Optional[int] = None,
        mask_multiplier: int = 255,
        dtype: str = "float32",
        vector: Optional[str] = None,
        simplify_tolerance: Optional[float] = None,
        **kwargs: Any,
    ) -> None:
        """Save the predicted mask to the output path.

        Args:
            output (str): The path to the output image.
            index (Optional[int]): The index of the mask to save.
            mask_multiplier (int): The mask multiplier for the output mask.
            dtype (str): The data type of the output image.
            vector (Optional[str]): The path to the output vector file.
            simplify_tolerance (Optional[float]): The maximum allowed geometry displacement.
            **kwargs (Any): Additional keyword arguments.
        """
        if self.scores is None:
            raise ValueError("No predictions found. Please run predict() first.")

        if index is None:
            index = self.scores.argmax(axis=0)

        array = self.masks[index] * mask_multiplier
        self.prediction = array
        common.array_to_image(array, output, self.source, dtype=dtype, **kwargs)

        if vector is not None:
            common.raster_to_vector(
                output, vector, simplify_tolerance=simplify_tolerance
            )

    def show_masks(
        self,
        figsize: Tuple[int, int] = (12, 10),
        cmap: str = "tab20",
        axis: str = "off",
        unique: bool = True,
        **kwargs: Any,
    ) -> None:
        """Show the binary mask or the mask of objects with unique values.

        Args:
            figsize (tuple): The figure size.
            cmap (str): The colormap. Default is "tab20" for showing unique objects.
                Use "binary_r" for binary masks when unique=False.
                Other good options: "viridis", "nipy_spectral", "rainbow".
            axis (str): Whether to show the axis.
            unique (bool): If True, each mask gets a unique color value. If False, binary mask.
            **kwargs: Additional keyword arguments passed to save_masks() for filtering
                (e.g., min_size, max_size, dtype).
        """

        # Always regenerate mask array to ensure it matches the unique parameter
        # This prevents showing stale cached binary masks when unique=True is requested
        self.save_masks(output=None, unique=unique, **kwargs)

        if self.objects is None:
            # save_masks would have printed a message if no masks met criteria
            return

        plt.figure(figsize=figsize)
        plt.imshow(self.objects, cmap=cmap, interpolation="nearest")
        plt.axis(axis)

        plt.show()

    def show_anns(
        self,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
        show_bbox: bool = True,
        show_score: bool = True,
        output: Optional[str] = None,
        blend: bool = True,
        alpha: float = 0.5,
        font_scale: float = 0.8,
        **kwargs: Any,
    ) -> None:
        """Show the annotations (objects with random color) on the input image.

        This method uses OpenCV for fast rendering, which is significantly faster
        than matplotlib when there are many objects to plot.

        Args:
            figsize (tuple): The figure size (used for display).
            axis (str): Whether to show the axis.
            show_bbox (bool): Whether to show the bounding box.
            show_score (bool): Whether to show the score.
            output (str, optional): The path to the output image. If provided, the
                figure will be saved instead of displayed.
            blend (bool): Whether to show the input image as background. If False,
                only annotations will be shown on a white background.
            alpha (float): The alpha value for the annotations.
            font_scale (float): The font scale for labels. Defaults to 0.8.
            **kwargs: Additional keyword arguments (kept for backward compatibility).
        """

        if self.image is None:
            print("Please run set_image() first.")
            return

        if self.masks is None or len(self.masks) == 0:
            return

        # Create the blended image using OpenCV (much faster than matplotlib)
        blended = self._render_anns_opencv(
            show_bbox=show_bbox,
            show_score=show_score,
            blend=blend,
            alpha=alpha,
            font_scale=font_scale,
        )

        if output is not None:
            # Save directly using OpenCV
            cv2.imwrite(output, cv2.cvtColor(blended, cv2.COLOR_RGB2BGR))
            print(f"Saved annotations to {output}")
        else:
            # Display the image
            self._display_image(blended, figsize=figsize, axis=axis)

    def _render_anns_opencv(
        self,
        show_bbox: bool = True,
        show_score: bool = True,
        blend: bool = True,
        alpha: float = 0.5,
        font_scale: float = 0.8,
    ) -> np.ndarray:
        """Render annotations using OpenCV for fast performance.

        Args:
            show_bbox (bool): Whether to show the bounding box.
            show_score (bool): Whether to show the score.
            blend (bool): Whether to show the input image as background.
            alpha (float): The alpha value for the annotations.
            font_scale (float): The font scale for labels.

        Returns:
            np.ndarray: The rendered image as RGB numpy array.
        """
        # Get image dimensions
        h, w = self.image.shape[:2]

        # Create base image
        if blend:
            frame_np = self.image.astype(np.float32)
        else:
            frame_np = np.ones((h, w, 3), dtype=np.float32) * 255

        nb_objects = len(self.scores)

        # Use the same color generation as the original method for consistency
        COLORS = generate_colors(n_colors=128, n_samples=5000)
        # Convert from 0-1 float RGB to 0-255 int RGB for OpenCV
        colors_rgb = [
            (int(c[0] * 255), int(c[1] * 255), int(c[2] * 255)) for c in COLORS
        ]

        # Create overlay for all masks
        overlay = np.zeros((h, w, 3), dtype=np.float32)
        mask_combined = np.zeros((h, w), dtype=np.float32)
        labels_to_draw = []

        for i in range(nb_objects):
            color = colors_rgb[i % len(colors_rgb)]

            # Handle both tensor and numpy array formats
            mask = self.masks[i]
            if hasattr(mask, "cpu"):
                mask_np = mask.squeeze().cpu().numpy()
            elif hasattr(mask, "numpy"):
                mask_np = mask.squeeze().numpy()
            else:
                mask_np = np.squeeze(mask)

            # Ensure mask is 2D
            if mask_np.ndim > 2:
                mask_np = mask_np[0]

            # Resize mask if it doesn't match frame dimensions
            if mask_np.shape != (h, w):
                mask_np = cv2.resize(
                    mask_np.astype(np.float32),
                    (w, h),
                    interpolation=cv2.INTER_NEAREST,
                )

            # Add color to overlay where mask is present
            mask_bool = mask_np > 0
            for c in range(3):
                overlay[:, :, c] = np.where(mask_bool, color[c], overlay[:, :, c])
            mask_combined = np.maximum(mask_combined, mask_np.astype(np.float32))

            # Collect label info for bounding boxes
            if show_bbox and self.boxes is not None:
                # Handle score extraction
                score = self.scores[i]
                if hasattr(score, "item"):
                    prob = score.item()
                else:
                    prob = float(score)

                if show_score:
                    text = f"(id={i}, {prob=:.2f})"
                else:
                    text = f"(id={i})"

                # Handle box extraction
                box = self.boxes[i]
                if hasattr(box, "cpu"):
                    box = box.cpu().numpy()
                elif hasattr(box, "numpy"):
                    box = box.numpy()
                else:
                    box = np.array(box)

                labels_to_draw.append((box, text, color))

        # Blend overlay with frame
        mask_3d = mask_combined[:, :, np.newaxis]
        blended = frame_np * (1 - mask_3d * alpha) + overlay * (mask_3d * alpha)
        blended = np.clip(blended, 0, 255).astype(np.uint8)

        # Draw bounding boxes and labels using OpenCV
        for box, text, color in labels_to_draw:
            x1, y1, x2, y2 = map(lambda v: int(round(v)), box[:4])
            # Clip bounding box coordinates to image boundaries
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(w, x2), min(h, y2)

            # Draw bounding box
            cv2.rectangle(blended, (x1, y1), (x2, y2), color, 2)

            # Get text size for background rectangle
            font = cv2.FONT_HERSHEY_SIMPLEX
            thickness = max(1, int(font_scale * 2.5))
            (text_w, text_h), baseline = cv2.getTextSize(
                text, font, font_scale, thickness
            )

            # Draw background rectangle for text
            pad = 2
            text_x = x1
            text_y = y1 - 5
            if text_y - text_h - pad < 0:
                text_y = y2 + text_h + 5

            # Semi-transparent background for text
            bg_x1 = text_x - pad
            bg_y1 = text_y - text_h - pad
            bg_x2 = text_x + text_w + pad
            bg_y2 = text_y + pad + baseline

            # Clip to image boundaries
            bg_x1, bg_y1 = max(0, bg_x1), max(0, bg_y1)
            bg_x2, bg_y2 = min(w, bg_x2), min(h, bg_y2)

            if bg_x2 > bg_x1 and bg_y2 > bg_y1:
                sub_img = blended[bg_y1:bg_y2, bg_x1:bg_x2].astype(np.float32)
                bg_color = np.array(color, dtype=np.float32)
                blend_rect = (sub_img * 0.3 + bg_color * 0.7).astype(np.uint8)
                blended[bg_y1:bg_y2, bg_x1:bg_x2] = blend_rect

            # Draw text
            cv2.putText(
                blended,
                text,
                (text_x, text_y),
                font,
                font_scale,
                (255, 255, 255),
                thickness,
                cv2.LINE_AA,
            )

        return blended

    def _display_image(
        self,
        image: np.ndarray,
        figsize: Tuple[int, int] = (12, 10),
        axis: str = "off",
    ) -> None:
        """Display an image, using IPython display if available for better performance.

        Args:
            image (np.ndarray): The image to display (RGB format).
            figsize (tuple): The figure size.
            axis (str): Whether to show the axis.
        """
        try:
            # Try to use IPython display for better notebook performance
            from IPython.display import display

            # Save to temporary file and display
            temp_dir = common.make_temp_dir()
            temp_path = os.path.join(temp_dir, "temp_anns.png")
            cv2.imwrite(temp_path, cv2.cvtColor(image, cv2.COLOR_RGB2BGR))

            # Display using PIL Image (works well in notebooks)
            img_display = Image.open(temp_path)

            # Resize for display based on figsize (assuming 100 DPI)
            display_width = figsize[0] * 100
            aspect_ratio = image.shape[0] / image.shape[1]
            display_height = int(display_width * aspect_ratio)
            img_display = img_display.resize(
                (display_width, display_height), Image.Resampling.LANCZOS
            )

            display(img_display)

        except ImportError:
            # Fall back to matplotlib for non-notebook environments
            plt.figure(figsize=figsize)
            plt.imshow(image)
            plt.axis(axis)
            plt.show()

    def raster_to_vector(
        self,
        raster: str,
        vector: str,
        simplify_tolerance: Optional[float] = None,
        **kwargs,
    ) -> None:
        """Convert a raster image file to a vector dataset.

        Args:
            raster (str): The path to the raster image.
            vector (str): The path to the output vector file.
            simplify_tolerance (float, optional): The maximum allowed geometry displacement.
        """
        common.raster_to_vector(
            raster, vector, simplify_tolerance=simplify_tolerance, **kwargs
        )

    def show_map(
        self,
        basemap="Esri.WorldImagery",
        out_dir=None,
        min_size=10,
        max_size=None,
        prompt="text",
        **kwargs,
    ):
        """Show the interactive map.

        Args:
            basemap (str, optional): The basemap. Valid options include "Esri.WorldImagery", "OpenStreetMap", "HYBRID", "ROADMAP", "TERRAIN", etc. See the leafmap documentation for a full list of supported basemaps.
            out_dir (str, optional): The path to the output directory. Defaults to None.
            min_size (int, optional): The minimum size of the object. Defaults to 10.
            max_size (int, optional): The maximum size of the object. Defaults to None.
            prompt (str, optional): The prompt type. Defaults to "text".
                Valid options include "text" and "point".

        Returns:
            leafmap.Map: The map object.
        """
        if prompt.lower() == "text":
            return common.text_sam_gui(
                self,
                basemap=basemap,
                out_dir=out_dir,
                box_threshold=self.confidence_threshold,
                text_threshold=self.mask_threshold,
                min_size=min_size,
                max_size=max_size,
                **kwargs,
            )
        elif prompt.lower() == "point":
            return common.sam_map_gui(
                self,
                basemap=basemap,
                out_dir=out_dir,
                min_size=min_size,
                max_size=max_size,
                **kwargs,
            )
        else:
            raise ValueError(f"Invalid prompt: {prompt}. Please use 'text' or 'point'.")

    def show_canvas(
        self,
        fg_color: Tuple[int, int, int] = (0, 255, 0),
        bg_color: Tuple[int, int, int] = (0, 0, 255),
        radius: int = 5,
    ) -> Tuple[list, list]:
        """Show a canvas to collect foreground and background points.

        Args:
            fg_color (Tuple[int, int, int]): The color for foreground points.
            bg_color (Tuple[int, int, int]): The color for background points.
            radius (int): The radius of the points.

        Returns:
            Tuple of foreground and background points.
        """

        if self.image is None:
            raise ValueError("Please run set_image() first.")

        image = self.image
        fg_points, bg_points = common.show_canvas(image, fg_color, bg_color, radius)
        self.fg_points = fg_points
        self.bg_points = bg_points
        point_coords = fg_points + bg_points
        point_labels = [1] * len(fg_points) + [0] * len(bg_points)
        self.point_coords = point_coords
        self.point_labels = point_labels

        return fg_points, bg_points

    def predict_inst(
        self,
        point_coords: Optional[Union[np.ndarray, List[List[float]]]] = None,
        point_labels: Optional[Union[np.ndarray, List[int]]] = None,
        box: Optional[Union[np.ndarray, List[float], List[List[float]]]] = None,
        mask_input: Optional[np.ndarray] = None,
        multimask_output: bool = True,
        return_logits: bool = False,
        normalize_coords: bool = True,
        point_crs: Optional[str] = None,
        box_crs: Optional[str] = None,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Predict masks for the given input prompts using SAM3's interactive instance
        segmentation (SAM1-style task). This enables point and box prompts for
        precise object segmentation.

        Note: This method requires the model to be initialized with
        `enable_inst_interactivity=True` (Meta backend only).

        Args:
            point_coords (np.ndarray or List, optional): A Nx2 array or list of point
                prompts to the model. Each point is in (X, Y) pixel coordinates.
                Can be a numpy array or a Python list like [[x1, y1], [x2, y2]].
            point_labels (np.ndarray or List, optional): A length N array or list of
                labels for the point prompts. 1 indicates a foreground point and
                0 indicates a background point.
            box (np.ndarray or List, optional): A length 4 array/list or Bx4 array/list
                of box prompt(s) to the model, in XYXY format.
            mask_input (np.ndarray, optional): A low resolution mask input to the
                model, typically coming from a previous prediction iteration.
                Has form 1xHxW (or BxHxW for batched), where H=W=256 for SAM.
            multimask_output (bool): If True, the model will return three masks.
                For ambiguous input prompts (such as a single click), this will
                often produce better masks than a single prediction. If only a
                single mask is needed, the model's predicted quality score can
                be used to select the best mask. For non-ambiguous prompts, such
                as multiple input prompts, multimask_output=False can give
                better results. Defaults to True.
            return_logits (bool): If True, returns un-thresholded mask logits
                instead of binary masks. Defaults to False.
            normalize_coords (bool): If True, the point coordinates will be
                normalized to the range [0, 1] and point_coords is expected to
                be w.r.t. image dimensions. Defaults to True.
            point_crs (str, optional): Coordinate reference system for point
                coordinates (e.g., "EPSG:4326"). Only used if the source image
                is a GeoTIFF. If None, points are in pixel coordinates.
            box_crs (str, optional): Coordinate reference system for box
                coordinates (e.g., "EPSG:4326"). Only used if the source image
                is a GeoTIFF. If None, box is in pixel coordinates.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray]:
                - masks: The output masks in CxHxW format (or BxCxHxW for batched
                  box input), where C is the number of masks, and (H, W) is the
                  original image size.
                - scores: An array of length C (or BxC) containing the model's
                  predictions for the quality of each mask.
                - logits: An array of shape CxHxW (or BxCxHxW), where C is the
                  number of masks and H=W=256. These low resolution logits can
                  be passed to a subsequent iteration as mask input.

        Example:
            >>> # Initialize with instance interactivity enabled
            >>> sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
            >>> sam.set_image("image.jpg")
            >>>
            >>> # Single point prompt
            >>> point_coords = np.array([[520, 375]])
            >>> point_labels = np.array([1])
            >>> masks, scores, logits = sam.predict_inst(
            ...     point_coords=point_coords,
            ...     point_labels=point_labels,
            ...     multimask_output=True,
            ... )
            >>>
            >>> # Select best mask based on score
            >>> best_mask_idx = np.argmax(scores)
            >>> best_mask = masks[best_mask_idx]
            >>>
            >>> # Refine with additional points
            >>> point_coords = np.array([[500, 375], [1125, 625]])
            >>> point_labels = np.array([1, 0])  # foreground and background
            >>> masks, scores, logits = sam.predict_inst(
            ...     point_coords=point_coords,
            ...     point_labels=point_labels,
            ...     mask_input=logits[best_mask_idx:best_mask_idx+1],  # Use previous best
            ...     multimask_output=False,
            ... )
            >>>
            >>> # Box prompt
            >>> box = np.array([425, 600, 700, 875])
            >>> masks, scores, logits = sam.predict_inst(box=box, multimask_output=False)
            >>>
            >>> # Combined box and point prompt
            >>> box = np.array([425, 600, 700, 875])
            >>> point_coords = np.array([[575, 750]])
            >>> point_labels = np.array([0])  # Exclude this region
            >>> masks, scores, logits = sam.predict_inst(
            ...     point_coords=point_coords,
            ...     point_labels=point_labels,
            ...     box=box,
            ...     multimask_output=False,
            ... )
        """
        if self.backend != "meta":
            raise NotImplementedError(
                "predict_inst is only available for the Meta backend. "
                "Please initialize with backend='meta' and enable_inst_interactivity=True."
            )

        if self.inference_state is None:
            raise ValueError("No image set. Please call set_image() first.")

        if (
            not hasattr(self.model, "inst_interactive_predictor")
            or self.model.inst_interactive_predictor is None
        ):
            raise ValueError(
                "Instance interactivity not enabled. Please initialize with "
                "enable_inst_interactivity=True."
            )

        # Convert lists to numpy arrays
        if point_coords is not None and not isinstance(point_coords, np.ndarray):
            point_coords = np.array(point_coords)
        if point_labels is not None and not isinstance(point_labels, np.ndarray):
            point_labels = np.array(point_labels)
        if box is not None and not isinstance(box, np.ndarray):
            box = np.array(box)

        # Transform point coordinates from CRS to pixel coordinates if needed
        if (
            point_coords is not None
            and point_crs is not None
            and self.source is not None
        ):
            point_coords = np.array(point_coords)
            point_coords, _ = common.coords_to_xy(
                self.source, point_coords, point_crs, return_out_of_bounds=True
            )

        # Transform box coordinates from CRS to pixel coordinates if needed
        if box is not None and box_crs is not None and self.source is not None:
            box = np.array(box)
            if box.ndim == 1:
                # Single box [xmin, ymin, xmax, ymax]
                xmin, ymin, xmax, ymax = box
                min_coords = np.array([[xmin, ymin]])
                max_coords = np.array([[xmax, ymax]])
                min_xy, _ = common.coords_to_xy(
                    self.source, min_coords, box_crs, return_out_of_bounds=True
                )
                max_xy, _ = common.coords_to_xy(
                    self.source, max_coords, box_crs, return_out_of_bounds=True
                )
                x1_px, y1_px = min_xy[0]
                x2_px, y2_px = max_xy[0]
                box = np.array(
                    [
                        min(x1_px, x2_px),
                        min(y1_px, y2_px),
                        max(x1_px, x2_px),
                        max(y1_px, y2_px),
                    ]
                )
            else:
                # Multiple boxes [B, 4]
                transformed_boxes = []
                for b in box:
                    xmin, ymin, xmax, ymax = b
                    min_coords = np.array([[xmin, ymin]])
                    max_coords = np.array([[xmax, ymax]])
                    min_xy, _ = common.coords_to_xy(
                        self.source, min_coords, box_crs, return_out_of_bounds=True
                    )
                    max_xy, _ = common.coords_to_xy(
                        self.source, max_coords, box_crs, return_out_of_bounds=True
                    )
                    x1_px, y1_px = min_xy[0]
                    x2_px, y2_px = max_xy[0]
                    transformed_boxes.append(
                        [
                            min(x1_px, x2_px),
                            min(y1_px, y2_px),
                            max(x1_px, x2_px),
                            max(y1_px, y2_px),
                        ]
                    )
                box = np.array(transformed_boxes)

        # Call the model's predict_inst method
        masks, scores, logits = self.model.predict_inst(
            self.inference_state,
            point_coords=point_coords,
            point_labels=point_labels,
            box=box,
            mask_input=mask_input,
            multimask_output=multimask_output,
            return_logits=return_logits,
            normalize_coords=normalize_coords,
        )

        # Store results
        self.masks = (
            [masks[i] for i in range(len(masks))] if masks.ndim > 2 else [masks]
        )
        self.scores = list(scores) if isinstance(scores, np.ndarray) else [scores]
        self.logits = logits

        return masks, scores, logits

    def show_inst_masks(
        self,
        masks: np.ndarray,
        scores: np.ndarray,
        point_coords: Optional[Union[np.ndarray, List[List[float]]]] = None,
        point_labels: Optional[Union[np.ndarray, List[int]]] = None,
        box_coords: Optional[Union[np.ndarray, List[float]]] = None,
        figsize: Tuple[int, int] = (10, 10),
        borders: bool = True,
    ) -> None:
        """
        Display masks from predict_inst results with optional point and box overlays.

        Args:
            masks (np.ndarray): Masks from predict_inst, shape CxHxW.
            scores (np.ndarray): Scores from predict_inst, shape C.
            point_coords (np.ndarray or List, optional): Point coordinates used for prompts.
                Can be a numpy array or a Python list like [[x1, y1], [x2, y2]].
            point_labels (np.ndarray or List, optional): Point labels (1=foreground, 0=background).
            box_coords (np.ndarray or List, optional): Box coordinates used for prompt.
                Can be a numpy array or a Python list like [x1, y1, x2, y2].
            figsize (Tuple[int, int]): Figure size for each mask display.
            borders (bool): Whether to draw contour borders on masks.

        Example:
            >>> sam = SamGeo3(backend="meta", enable_inst_interactivity=True)
            >>> sam.set_image("image.jpg")
            >>> masks, scores, logits = sam.predict_inst(
            ...     point_coords=[[520, 375]],
            ...     point_labels=[1],
            ... )
            >>> sam.show_inst_masks(
            ...     masks, scores,
            ...     point_coords=[[520, 375]],
            ...     point_labels=[1],
            ... )
        """
        if self.image is None:
            raise ValueError("No image set. Please call set_image() first.")

        # Convert lists to numpy arrays
        if point_coords is not None and not isinstance(point_coords, np.ndarray):
            point_coords = np.array(point_coords)
        if point_labels is not None and not isinstance(point_labels, np.ndarray):
            point_labels = np.array(point_labels)
        if box_coords is not None and not isinstance(box_coords, np.ndarray):
            box_coords = np.array(box_coords)

        # Sort by score (descending)
        sorted_ind = np.argsort(scores)[::-1]
        masks = masks[sorted_ind]
        scores = scores[sorted_ind]

        for i, (mask, score) in enumerate(zip(masks, scores)):
            fig = plt.figure(figsize=figsize)
            plt.imshow(self.image)

            # Show mask
            h, w = mask.shape[-2:]
            mask_uint8 = mask.astype(np.uint8)
            color = np.array([30 / 255, 144 / 255, 255 / 255, 0.6])
            mask_image = mask_uint8.reshape(h, w, 1) * color.reshape(1, 1, -1)

            if borders:
                contours, _ = cv2.findContours(
                    mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
                )
                contours = [
                    cv2.approxPolyDP(contour, epsilon=0.01, closed=True)
                    for contour in contours
                ]
                mask_image = cv2.drawContours(
                    mask_image, contours, -1, (1, 1, 1, 0.5), thickness=2
                )

            plt.gca().imshow(mask_image)

            # Show points if provided
            if point_coords is not None and point_labels is not None:
                pos_points = point_coords[point_labels == 1]
                neg_points = point_coords[point_labels == 0]
                plt.scatter(
                    pos_points[:, 0],
                    pos_points[:, 1],
                    color="green",
                    marker="*",
                    s=375,
                    edgecolor="white",
                    linewidth=1.25,
                )
                plt.scatter(
                    neg_points[:, 0],
                    neg_points[:, 1],
                    color="red",
                    marker="*",
                    s=375,
                    edgecolor="white",
                    linewidth=1.25,
                )

            # Show box if provided
            if box_coords is not None:
                x0, y0 = box_coords[0], box_coords[1]
                box_w, box_h = (
                    box_coords[2] - box_coords[0],
                    box_coords[3] - box_coords[1],
                )
                plt.gca().add_patch(
                    plt.Rectangle(
                        (x0, y0),
                        box_w,
                        box_h,
                        edgecolor="green",
                        facecolor=(0, 0, 0, 0),
                        lw=2,
                    )
                )

            if len(scores) > 1:
                plt.title(f"Mask {i+1}, Score: {score:.3f}", fontsize=18)

            plt.axis("off")
            plt.show()


def generate_colors(n_colors: int = 256, n_samples: int = 5000) -> np.ndarray:
    """Generate colors for the masks.

    Args:
        n_colors (int, optional): The number of colors to generate. Defaults to 256.
        n_samples (int, optional): The number of samples to generate. Defaults to 5000.

    Returns:
        np.ndarray: The generated colors in RGB format.
    """
    # Step 1: Random RGB samples
    np.random.seed(42)
    rgb = np.random.rand(n_samples, 3)
    # Step 2: Convert to LAB for perceptual uniformity
    # print(f"Converting {n_samples} RGB samples to LAB color space...")
    lab = rgb2lab(rgb.reshape(1, -1, 3)).reshape(-1, 3)
    # print("Conversion to LAB complete.")
    # Step 3: k-means clustering in LAB
    kmeans = KMeans(n_clusters=n_colors, n_init=10)
    # print(f"Fitting KMeans with {n_colors} clusters on {n_samples} samples...")
    kmeans.fit(lab)
    # print("KMeans fitting complete.")
    centers_lab = kmeans.cluster_centers_
    # Step 4: Convert LAB back to RGB
    colors_rgb = lab2rgb(centers_lab.reshape(1, -1, 3)).reshape(-1, 3)
    colors_rgb = np.clip(colors_rgb, 0, 1)
    return colors_rgb


def plot_bbox(
    img_height,
    img_width,
    box,
    box_format="XYXY",
    relative_coords=True,
    color="r",
    linestyle="solid",
    text=None,
    ax=None,
):
    """Plot the bounding box on the image.

    Args:
        img_height (int): The height of the image.
        img_width (int): The width of the image.
        box (np.ndarray): The bounding box.
        box_format (str): The format of the bounding box.
        relative_coords (bool): Whether the coordinates are relative to the image.
        color (str): The color of the bounding box.
        linestyle (str): The line style of the bounding box.
        text (str): The text to display in the bounding box.
        ax (matplotlib.axes.Axes, optional): The axis to plot the bounding box on.
    """
    # Convert box to numpy array if it's a tensor
    if hasattr(box, "numpy"):
        box = box.numpy()
    elif hasattr(box, "cpu"):
        box = box.cpu().numpy()

    if box_format == "XYXY":
        x, y, x2, y2 = box
        w = x2 - x
        h = y2 - y
    elif box_format == "XYWH":
        x, y, w, h = box
    elif box_format == "CxCyWH":
        cx, cy, w, h = box
        x = cx - w / 2
        y = cy - h / 2
    else:
        raise RuntimeError(f"Invalid box_format {box_format}")

    if relative_coords:
        x *= img_width
        w *= img_width
        y *= img_height
        h *= img_height

    if ax is None:
        ax = plt.gca()
    rect = patches.Rectangle(
        (float(x), float(y)),
        float(w),
        float(h),
        linewidth=1.5,
        edgecolor=color,
        facecolor="none",
        linestyle=linestyle,
    )
    ax.add_patch(rect)
    if text is not None:
        facecolor = "w"
        ax.text(
            float(x),
            float(y) - 5,
            text,
            color=color,
            weight="bold",
            fontsize=8,
            bbox={"facecolor": facecolor, "alpha": 0.75, "pad": 2},
        )


def draw_box_on_image(image, box, color=(0, 255, 0), thickness=2):
    """Draw a bounding box on an image.

    Args:
        image (PIL.Image.Image or np.ndarray): The image to draw on.
        box (List[float]): Bounding box in XYWH format [x, y, width, height].
        color (Tuple[int, int, int]): RGB color for the box. Default is green.
        thickness (int): Line thickness in pixels.

    Returns:
        PIL.Image.Image: Image with box drawn.
    """
    from PIL import ImageDraw

    # Convert numpy array to PIL Image if needed
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)

    # Make a copy to avoid modifying the original
    image_copy = image.copy()
    draw = ImageDraw.Draw(image_copy)

    # Extract box coordinates (XYWH format)
    x, y, w, h = box

    # Draw rectangle
    draw.rectangle([x, y, x + w, y + h], outline=color, width=thickness)

    return image_copy


def plot_mask(mask, color="r", alpha=0.5, ax=None):
    """Plot the mask on the image.

    Args:
        mask (np.ndarray): The mask to plot.
        color (str): The color of the mask.
        ax (matplotlib.axes.Axes, optional): The axis to plot the mask on.
        alpha (float): The alpha value for the mask.
    """
    im_h, im_w = mask.shape
    mask_img = np.zeros((im_h, im_w, 4), dtype=np.float32)
    mask_img[..., :3] = to_rgb(color)
    mask_img[..., 3] = mask * alpha
    # Use the provided ax or the current axis
    if ax is None:
        ax = plt.gca()
    ax.imshow(mask_img)

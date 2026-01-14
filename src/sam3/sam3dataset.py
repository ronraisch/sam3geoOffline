import logging
from pathlib import Path
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from typing import TypedDict, Optional, List, Union, Callable, Dict, Any

logger = logging.getLogger(__name__)

class DataItem(TypedDict, total=False):
    image: Image.Image  # Dataset always outputs PIL Images
    id: Union[int, str]
    prompt: Optional[str]
    boxes: Optional[List[List[float]]]

class SAM3Dataset(Dataset):
    """
    Dataset for SAM3 that only accepts str/Path and numpy arrays as inputs.
    Outputs PIL Images for processing.
    """
    def __init__(
        self,
        images: List[Union[str, Path, np.ndarray]],
        prompts: Optional[Union[str, List[str]]] = None,
        input_boxes: Optional[List[List[List[float]]]] = None,
        id_func: Optional[Callable[[Union[str, Path, np.ndarray]], Union[int, str]]] = None,
        image_loader: Optional[Callable[[Union[str, Path, np.ndarray]], Image.Image]] = None
    ) -> None:
        self.images = images
        self.prompts = prompts
        self.input_boxes = input_boxes
        self.id_func = id_func
        self.image_loader = image_loader
        logger.debug(f"SAM3Dataset initialized with {len(images)} images")

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> DataItem:
        try:
            image_data = self.images[idx]
            logger.debug(f"Loading item {idx}: type={type(image_data)}")
            
            # Validate input type
            if not isinstance(image_data, (str, Path, np.ndarray)):
                raise TypeError(
                    f"Unsupported image type: {type(image_data)}. "
                    f"Expected str, Path, or np.ndarray, got {type(image_data)}"
                )
            
            # Use provided image loader or default behavior
            if self.image_loader is not None:
                image = self.image_loader(image_data)
            else:
                # Fallback: simple PIL loading (won't handle GeoTIFFs properly)
                if isinstance(image_data, (str, Path)):
                    image = Image.open(str(image_data)).convert("RGB")
                elif isinstance(image_data, np.ndarray):
                    image = Image.fromarray(image_data).convert("RGB")
                else:
                    raise TypeError(f"Unsupported image type: {type(image_data)}")
            
            img_id = self.id_func(image_data) if self.id_func else idx
            logger.debug(f"Item {idx} loaded, ID: {img_id}, Image size: {image.size}")

            item: DataItem = {
                "image": image,
                "id": img_id
            }

            if self.prompts is not None:
                prompt_val = self.prompts[idx] if isinstance(self.prompts, list) else self.prompts
                item["prompt"] = prompt_val
                logger.debug(f"Item {idx}: Added prompt: {prompt_val[:50] if isinstance(prompt_val, str) else prompt_val}")

            if self.input_boxes is not None:
                item["boxes"] = self.input_boxes[idx]
                logger.debug(f"Item {idx}: Added {len(self.input_boxes[idx])} boxes")

            return item
        except Exception as e:
            logger.error(f"Error loading item {idx}: {e}", exc_info=True)
            raise

def collate_fn(batch: List[DataItem]) -> Dict[str, List[Any]]:
    """Custom collate to handle non-tensor data like PIL images or strings."""
    if not batch:
        raise ValueError("Batch is empty")
    logger.debug(f"Collating batch of size {len(batch)}")
    result = {key: [d[key] for d in batch] for key in batch[0]}
    logger.debug(f"Collated keys: {list(result.keys())}")
    return result
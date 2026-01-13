import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np

class SAM3Dataset(Dataset):
    def __init__(self, images, prompts=None, input_boxes=None, id_func=None):
        self.images = images
        self.prompts = prompts
        self.input_boxes = input_boxes
        self.id_func = id_func

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        image_data = self.images[idx]
        img_id = self.id_func(image_data) if self.id_func else idx
        
        # Logic to load image if it's a path
        if isinstance(image_data, str):
            image = Image.open(image_data).convert("RGB")
        else:
            image = image_data

        item = {
            "image": image,
            "id": img_id
        }

        if self.prompts is not None:
            item["prompt"] = self.prompts[idx] if isinstance(self.prompts, list) else self.prompts
        
        if self.input_boxes is not None:
            item["boxes"] = self.input_boxes[idx]

        return item

def collate_fn(batch):
    """Custom collate to handle non-tensor data like PIL images or strings."""
    return {key: [d[key] for d in batch] for key in batch[0]}
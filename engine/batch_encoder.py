from typing import List, Optional, Tuple

import clip
import torch
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import Dataset


class ImageDataset(Dataset):
    def __init__(self, image_paths, transform):
        self.image_paths = image_paths
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)
        return image, image_path


# class BatchFeatureExtractor:
#     def __init__(self, modelname, batch_size=64, num_workers=4, device=None):
#         self.device = (
#             device if device else ("cuda" if torch.cuda.is_available() else "cpu")
#         )
#         self.batch_size = batch_size
#         self.num_workers = num_workers  # Optimized num_workers

#         # Print the settings when the module runs
#         print(f"🔥 Using device: {self.device.upper()}")
#         print(f"🛠 num_workers: {self.num_workers}")
#         print(f"📦 Batch size: {self.batch_size}")

#         # Load the pre-trained model and move to GPU
#         self.model = timm.create_model(modelname, pretrained=True, num_classes=0).to(
#             self.device
#         )
#         self.model.eval()

#         config = timm.data.resolve_model_data_config(model=modelname)
#         self.preprocess = timm.data.create_transform(**config, is_training=False)


class CLIPBatchFeatureExtractor:
    def __init__(
        self,
        model_name: str = "ViT-B/32",
        batch_size: int = 64,
        num_workers: int = 4,
        device: Optional[str] = None,
    ):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Print the settings when the module runs
        print(f"🔥 Using device: {self.device.upper()}")
        print(f"🛠 num_workers: {self.num_workers}")
        print(f"📦 Batch size: {self.batch_size}")

        # Load CLIP model
        self.model, self.preprocess = clip.load(model_name, device=self.device)
        self.model.eval()

    def extract_features(self, images) -> torch.Tensor:
        """
        Extract features from images using CLIP
        Args:
            images: Either a list of PIL images or a batch tensor
        Returns:
            Tensor of image features
        """
        with torch.no_grad():
            if isinstance(images, (list, tuple)):
                # If we get a list of PIL images, preprocess them
                processed_images = torch.stack([self.preprocess(img) for img in images])
            else:
                # If we already have a batch tensor
                processed_images = images

            # Move to device and get features
            processed_images = processed_images.to(self.device)
            image_features = self.model.encode_image(processed_images)
            # Normalize features
            image_features = F.normalize(image_features, dim=-1)

            return image_features.cpu()

    def encode_text(self, texts: List[str]) -> torch.Tensor:
        """
        Encode text queries using CLIP
        Args:
            texts: List of text queries
        Returns:
            Tensor of text features
        """
        with torch.no_grad():
            text_tokens = clip.tokenize(texts).to(self.device)
            text_features = self.model.encode_text(text_tokens)
            text_features = F.normalize(text_features, dim=-1)
        return text_features.cpu()

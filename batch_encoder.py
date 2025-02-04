
import timm
import torch
from PIL import Image
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform
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


class BatchFeatureExtractor:
    def __init__(self, modelname, batch_size=64, num_workers=4, device=None):
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.batch_size = batch_size
        self.num_workers = num_workers  # Optimized num_workers

        # Print the settings when the module runs
        print(f"🔥 Using device: {self.device.upper()}")
        print(f"🛠 num_workers: {self.num_workers}")
        print(f"📦 Batch size: {self.batch_size}")

        # Load the pre-trained model and move to GPU
        self.model = timm.create_model(
            modelname, pretrained=True, num_classes=0, global_pool="avg"
        ).to(self.device)
        self.model.eval()

        # Get the preprocessing function provided by TIMM for the model
        config = resolve_data_config({}, model=modelname)
        self.preprocess = create_transform(**config)

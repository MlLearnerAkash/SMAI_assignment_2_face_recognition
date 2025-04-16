from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        own_dir = os.path.join(root_dir, 'own')
        other_dir = os.path.join(root_dir, 'other', 'Humans')

        # Label 0 for 'own' (aggregate from all subfolders)
        for subdir, _, files in os.walk(own_dir):
            for file in files:
                if file.lower().endswith(('png', 'jpg', 'jpeg')):
                    self.image_paths.append(os.path.join(subdir, file))
                    self.labels.append(0)

        # Label 1 for 'other/Humans'
        for file in os.listdir(other_dir):
            if file.lower().endswith(('png', 'jpg', 'jpeg')):
                self.image_paths.append(os.path.join(other_dir, file))
                self.labels.append(1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os

class CustomBinaryDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        own_dir = os.path.join(root_dir, '')
        # other_dir = os.path.join(root_dir, 'other', 'Humans')

        # Label 0 for 'own' (aggregate from all subfolders)
        # for subdir, _, files in os.walk(own_dir):
        #     for file in files:
        #         if file.lower().endswith(('png', 'jpg', 'jpeg')):
        #             self.image_paths.append(os.path.join(subdir, file))
        #             self.labels.append(0)

        # # Label 1 for 'other/Humans'
        # for file in os.listdir(other_dir):
        #     if file.lower().endswith(('png', 'jpg', 'jpeg')):
        #         self.image_paths.append(os.path.join(other_dir, file))
        #         self.labels.append(1)

        subdirs = sorted([
        d for d in os.listdir(own_dir) 
        if os.path.isdir(os.path.join(own_dir, d))
        ])

        # Assign unique label to each subdirectory
        for label, subdir_name in enumerate(subdirs):
            subdir_path = os.path.join(own_dir, subdir_name)
            
            # Walk through all files in this subdirectory (including nested subdirs)
            for root, _, files in os.walk(subdir_path):
                for file in files:
                    if file.lower().endswith(('png', 'jpg', 'jpeg')):
                        self.image_paths.append(os.path.join(root, file))
                        self.labels.append(label)  # Unique per subdirectory

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            image = self.transform(image)
        label = self.labels[idx]
        return image, label

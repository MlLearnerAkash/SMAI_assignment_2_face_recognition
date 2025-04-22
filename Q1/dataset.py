import os
import random
from torch.utils.data import Dataset
from PIL import Image

class CustomBinaryDataset(Dataset):
    def __init__(self, root_dir, total_per_class=100, transform=None):
        self.transform = transform
        self.image_paths = []
        self.labels = []

        # helper: sample total_per_class images from class_dir,
        # equally across its immediate sub‑directories (or the dir itself)
        def _sample_class(class_dir, label):
            # find immediate sub‑dirs (ignore files)
            subdirs = [d for d in os.listdir(class_dir)
                       if os.path.isdir(os.path.join(class_dir, d))]
            if not subdirs:
                # no sub‑folders → just sample from class_dir
                subdirs = ['.']

            n_subdirs = len(subdirs)
            base_n = total_per_class // n_subdirs
            extra = total_per_class % n_subdirs

            paths = []
            for i, sub in enumerate(subdirs):
                folder = class_dir if sub == '.' else os.path.join(class_dir, sub)
                all_imgs = [os.path.join(folder, f)
                            for f in os.listdir(folder)
                            if f.lower().endswith(('png','jpg','jpeg'))]
                k = base_n + (1 if i < extra else 0)
                if len(all_imgs) <= k:
                    # if not enough images, take them all (you could also sample with replacement)
                    picks = all_imgs
                else:
                    picks = random.sample(all_imgs, k)
                paths.extend(picks)

            # attach to dataset lists
            for p in paths:
                self.image_paths.append(p)
                self.labels.append(label)

        # sample 0='own'
        _sample_class(os.path.join(root_dir, 'own'), label=0)
        # sample 1='other'
        _sample_class(os.path.join(root_dir, 'other'), label=1)

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, self.labels[idx]

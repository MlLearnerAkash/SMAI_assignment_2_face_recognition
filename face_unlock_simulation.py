import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import argparse
import time
import os
import numpy as np
import wandb
from dataset import CustomBinaryDataset




def face_unlock(image, model):
    pass






if __name__ == "__main__":
    
    device = torch.device("cuda" if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')



    dataset_root = "/home/akash/ws/personal/SMAI_assignments/assignment_2/dataset"
    test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])
    dataset = CustomBinaryDataset(dataset_root, transform=test_transform)

    seed = 42
    generator = torch.Generator().manual_seed(seed)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    _, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)


    val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False
)

    model_chkpt = "/home/akash/ws/personal/SMAI_assignments/assignment_2/output/best_model.pth"
    model = models.resnet18(pretrained=False)

    model.load_state_dict(torch.load(model_chkpt))


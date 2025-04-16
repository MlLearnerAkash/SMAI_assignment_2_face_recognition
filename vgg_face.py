import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms, datasets
from torch.utils.data import DataLoader, random_split
import argparse
import time
import os
import numpy as np
import wandb
from sklearn.metrics import classification_report
from vgg_face_model import vgg_face_dag
from dataset import CustomBinaryDataset
# Custom VGG-Face model loading (you'll need to provide the actual PyTorch VGG-Face implementation)
# from vgg_face import VGG_FACE  # Replace with actual VGG-Face implementation

# Argument parsing
parser = argparse.ArgumentParser(description='VGG-Face Training Script')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to dataset directory')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--output_dir', type=str, default='output',
                    help='Output directory for models')
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                    help='Device to use for training (cuda/cpu)')
args = parser.parse_args()

# Device configuration
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')

# Initialize Weights & Biases
wandb.init(project='vgg-face-classification', config=args)
config = wandb.config

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# VGG-Face specific preprocessing parameters
VGG_MEAN = [129.1863, 104.7624, 93.5940]

class BGRTransform(object):
    """Convert RGB to BGR"""
    def __call__(self, img):
        return img[[2,1,0],:,:]

class Scale255(object):
    """Scale image to 0-255 range"""
    def __call__(self, tensor):
        return tensor * 255

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    # Scale255(),
    transforms.Normalize(mean=VGG_MEAN, std=[1,1,1]),  # Original code didn't use std
    BGRTransform(),
    transforms.ToPILImage(),
    transforms.ToTensor()
])

val_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    # Scale255(),
    transforms.Normalize(mean=VGG_MEAN, std=[1,1,1]),
    BGRTransform(),
    transforms.ToPILImage(),
    transforms.ToTensor()
])



# Initialize VGG-Face model
def initialize_model():
    # model = torch.load('/home/akash/ws/personal/SMAI_assignments/assignment_2/vgg_face_dag.pth',
    #                    )  # Replace with actual initialization
    # # Modify last layer for binary classification
    model  = vgg_face_dag("/home/akash/ws/personal/SMAI_assignments/assignment_2/vgg_face_dag.pth")
    model.fc8 = nn.Linear(4096, 2)  # Adjust based on actual architecture
    return model

model = initialize_model()#.to(device)
wandb.watch(model)

# Dataset and dataloaders
dataset = CustomBinaryDataset(args.data_dir, transform=train_transform)

# Split dataset
seed = 42
generator = torch.Generator().manual_seed(seed)
train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True
)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

def train_epoch(model, loader):
    model.train()
    running_loss = 0.0
    running_corrects = 0
    
    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        running_corrects += torch.sum(preds == labels.data)
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc

def evaluate(model, loader):
    model.eval()
    running_loss = 0.0
    running_corrects = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            running_corrects += torch.sum(preds == labels.data)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / len(loader.dataset)
    epoch_acc = running_corrects.double() / len(loader.dataset)
    return epoch_loss, epoch_acc, all_preds, all_labels

# Training loop and rest of the code remains similar to previous implementation...
# (Include the training loop and W&B logging from previous implementation)
def train_model():
    best_acc = 0.0
    train_losses = []
    val_losses = []
    
    for epoch in range(args.epochs):
        print(f'Epoch {epoch+1}/{args.epochs}')
        
        # Training phase
        train_loss, train_acc = train_epoch(model, train_loader)
        
        # Validation phase
        val_loss, val_acc, _, _ = evaluate(model, val_loader)
        
        # Update learning rate
        scheduler.step()
        
        # Save losses for plotting
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # Log metrics to wandb
        wandb.log({
            'epoch': epoch,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'val_loss': val_loss,
            'val_acc': val_acc,
            'lr': scheduler.get_last_lr()[0]
        })
        
        print(f'Train Loss: {train_loss:.4f} Acc: {train_acc:.4f}')
        print(f'Val Loss: {val_loss:.4f} Acc: {val_acc:.4f}\n')
        
        # Save best model
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), os.path.join(args.output_dir, 'best_model.pth'))
    
    # Plot loss curves
    wandb.log({
        "train_val_loss_curve": wandb.plot.line_series(
            xs=np.arange(args.epochs),
            ys=[train_losses, val_losses],
            keys=["Train Loss", "Validation Loss"],
            title="Training/Validation Loss Curves",
            xname="Epoch"
        )
    })
    
    return train_losses, val_losses

def test_model(model, loader):
    loss, acc, preds, labels = evaluate(model, loader)
    report = classification_report(
        labels, preds,
        target_names=["own", "other"],
        output_dict=True
    )
    return report
if __name__ == '__main__':
    # Training and evaluation code similar to previous version
    # Remember to adjust for VGG-Face specific requirements
    start_time = time.time()
    
    # Train the model
    train_losses, val_losses = train_model()
    
    # Load best model for testing
    model.load_state_dict(torch.load(os.path.join(args.output_dir, 'best_model.pth')))
    test_report = test_model(model, val_loader)
    
    # Log test results
    wandb.log({
        'test_accuracy': test_report['accuracy'],
        'test_precision': test_report['weighted avg']['precision'],
        'test_recall': test_report['weighted avg']['recall'],
        'test_f1': test_report['weighted avg']['f1-score']
    })
    
    # Print final results
    print("\nTest Set Metrics:")
    print(f"Accuracy: {test_report['accuracy']:.4f}")
    print(f"Precision: {test_report['weighted avg']['precision']:.4f}")
    print(f"Recall: {test_report['weighted avg']['recall']:.4f}")
    print(f"F1-Score: {test_report['weighted avg']['f1-score']:.4f}")
    
    print(f'\nTraining completed in {(time.time() - start_time)//60:.0f}m {(time.time() - start_time)%60:.0f}s')
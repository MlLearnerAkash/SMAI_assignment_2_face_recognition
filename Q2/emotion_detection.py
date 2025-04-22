import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models, transforms, datasets
from torch.utils.data import DataLoader, random_split
import argparse
import time
import os
import matplotlib.pyplot as plt
import numpy as np
import wandb
import itertools
from sklearn.metrics import classification_report,confusion_matrix

from dataset import CustomBinaryDataset
from vgg_face_model import vgg_face_dag
# Argument parsing
parser = argparse.ArgumentParser(description='ResNet18 Training Script')
parser.add_argument('--data_dir', type=str, required=True,
                    help='Path to dataset directory')
parser.add_argument('--initialization', type=str, required=True,
                    choices=['pretrained', 'random'],
                    help='Model initialization: pretrained or random')
parser.add_argument('--epochs', type=int, default=10,
                    help='Number of training epochs')
parser.add_argument('--batch_size', type=int, default=32,
                    help='Batch size for training')
parser.add_argument('--lr', type=float, default=0.001,
                    help='Learning rate')
parser.add_argument('--output_dir', type=str, default='output',
                    help='Output directory for models')
parser.add_argument('--split_size', type=float, default=0.1)
parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
                    help='Device to use for training (cuda/cpu)')
parser.add_argument('--mode', default = "vgg_face", type= str,
                    help = "vgg face or resnet18")
args = parser.parse_args()

# Device configuration
device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
if args.device == 'cuda' and not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU instead")

name = f"{args.mode}_{args.initialization}_{args.lr}_{device.type}-2"
# Initialize Weights & Biases
wandb.init(project='emotion-recognition', config=args, name=name)
config = wandb.config

# Create output directory
os.makedirs(args.output_dir, exist_ok=True)

# Initialize ResNet18
def initialize_model(pretrained=True):
    
    if args.mode == "vgg_face":
        model  = vgg_face_dag("/home/akash/ws/personal/SMAI_assignments/assignment_2/vgg_face_dag.pth")
        model.fc8 = nn.Linear(4096, 3)  # Adjust based on actual architecture
        # for name, param in model.named_parameters():
        #     if not name.startswith('fc8'):
        #         param.requires_grad = False
    else:
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)

    return model

model = initialize_model(args.initialization == 'pretrained').to(device)
wandb.watch(model)

# Data transformations
train_transform = transforms.Compose([
    transforms.Resize((224,224)),
    # transforms.RandomResizedCrop(224),
    # transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(10),
    transforms.ColorJitter(),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# Dataset and dataloaders
dataset_root = args.data_dir
dataset = CustomBinaryDataset(dataset_root, transform=train_transform)

seed = 42
generator = torch.Generator().manual_seed(seed)
train_size = int(0.7 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=4,
    pin_memory=True if device.type == 'cuda' else False
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
def plot_confusion_matrix(cm, classes,
                            normalize=True,

                            title='Confusion matrix',
                            cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        
        plt.figure(figsize=(8, 6))
        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)
        
        fmt = '.2f' if normalize else 'd'
        thresh = cm.max() / 2.
        
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt),
                    horizontalalignment="center",
                    color="white" if cm[i, j] > thresh else "black")
        
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        return plt.gcf()

def test_model(model, loader):
    loss, acc, preds, labels = evaluate(model, loader)
    report = classification_report(
        labels, preds,
        target_names=["happy", "neutral", "sad"],
        output_dict=True
    )
    # 2) compute raw confusion matrix
    cm = confusion_matrix(labels, preds)

    # 3) log it to W&B as a chart
    #    wandb.plot.confusion_matrix returns a Plotly figure
    cm_plot = wandb.plot.confusion_matrix(
        probs=None,
        y_true=labels,
        preds=preds,
        class_names=["happy", "neutral", "sad"]
    )
    cm_fig = plot_confusion_matrix(cm, classes=["happy", "neutral", "sad"], title="Confusion Matrix")
    wandb.log({"confusion_matrix": cm_plot})
    wandb.log({"confusion matrix": wandb.Image(cm_fig)})
    return report

if __name__ == '__main__':
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
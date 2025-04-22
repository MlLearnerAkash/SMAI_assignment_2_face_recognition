import os
import torch
from PIL import Image, ImageDraw, ImageFont
import imageio

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
from sklearn.metrics import classification_report

from dataset import CustomBinaryDataset
from vgg_face_model import Vgg_face_dag

parser = argparse.ArgumentParser(description='Training & Inference Script for 3-Class Classification')
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
                    help='Device to use for training/inference')
parser.add_argument('--mode', default='resnet18', type=str,
                    help='Model type: vgg_face or resnet18')
args = parser.parse_args()

def initialize_model(pretrained=False):
    if args.mode == 'vgg_face':
        model = Vgg_face_dag()
        model.fc8 = nn.Linear(4096, 3)
    else:
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 3)
    return model

# Device setup
device = torch.device(args.device if torch.cuda.is_available() and args.device=='cuda' else 'cpu')
if args.device=='cuda' and not torch.cuda.is_available():
    print("Warning: CUDA not available, using CPU instead")

# Transforms
test_transform = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485,0.456,0.406], [0.229,0.224,0.225])
])

# Dataset & DataLoader
dataset = CustomBinaryDataset(args.data_dir, transform=test_transform)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_ds, val_ds = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(42))
val_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=4, pin_memory=(device.type=='cuda'))


# def inference_and_create_gif(model, loader, device, class_idx, class_name, gif_path, max_samples=20):
#     model.eval()
#     frames = []
#     count = 0

#     # Load font
#     try:
#         font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=32)
#     except IOError:
#         font = ImageFont.load_default()
#     scale = 3

#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)

#             for i in range(inputs.size(0)):
#                 if labels[i].item() != class_idx:
#                     continue

#                 # Unnormalize & to PIL
#                 img = inputs[i].cpu()
#                 mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
#                 std = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
#                 img = img*std + mean
#                 arr = (img.clamp(0,1).permute(1,2,0).numpy()*255).astype('uint8')
#                 pil_img = Image.fromarray(arr)

#                 # Resize & canvas
#                 w,h = pil_img.size
#                 w2,h2 = w*scale, h*scale
#                 pil_img = pil_img.resize((w2,h2), Image.BILINEAR)
#                 canvas = Image.new('RGB',(w2*2,h2),(255,255,255))
#                 canvas.paste(pil_img,(0,0))
#                 canvas.paste(pil_img,(w2,0))

#                 draw = ImageDraw.Draw(canvas)
#                 # GT & Pred
#                 draw.text((5,5), f"GT: {class_name}", fill='black', font=font)
#                 is_correct = (preds[i].item()==class_idx)
#                 color = 'green' if is_correct else 'red'
#                 pred_label = class_name if is_correct else str(preds[i].item())
#                 tw,th = draw.textsize(f"Pred: {pred_label}", font=font)
#                 draw.text((w2*2 - tw -5, 5), f"Pred: {pred_label}", fill=color, font=font)

#                 frames.append(canvas)
#                 count += 1
#                 if count>=max_samples: break
#             if count>=max_samples: break

#     os.makedirs(os.path.dirname(gif_path), exist_ok=True)
#     imageio.mimsave(gif_path, frames, duration=500)
#     print(f"Saved GIF for class '{class_name}' at: {gif_path}")


# if __name__=='__main__':
#     # Load model
#     model = initialize_model(pretrained=True).to(device)
#     ckpt = os.path.join(args.output_dir,'best_model.pth')
#     model.load_state_dict(torch.load(ckpt, map_location=device))

#     # Generate GIFs for each of the three classes
#     class_names = ['happy','neutral','sad']
#     for idx, name in enumerate(class_names):
#         gif_file = os.path.join(args.output_dir, f"{name}_faces.gif")
#         inference_and_create_gif(model, val_loader, device, idx, name, gif_file)
#=========================================================================================
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

def inference_and_create_gif_combined(model, loader, device, class_names, gif_path, max_samples_per_class=10):
    model.eval()
    frames = []
    all_preds = []
    all_labels = []
    print(">>>>>",class_names)
    class_sample_count = {i: 0 for i in range(len(class_names))}

    # Load font
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=32)
    except IOError:
        font = ImageFont.load_default()
    scale = 3

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            for i in range(inputs.size(0)):
                label = labels[i].item()
                pred = preds[i].item()

                if class_sample_count[label] >= max_samples_per_class:
                    continue

                all_preds.append(pred)
                all_labels.append(label)
                class_sample_count[label] += 1

                # Unnormalize & convert to PIL
                img = inputs[i].cpu()
                mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
                std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
                img = img * std + mean
                arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).astype('uint8')
                pil_img = Image.fromarray(arr)

                # Resize & canvas
                w, h = pil_img.size
                w2, h2 = w * scale, h * scale
                pil_img = pil_img.resize((w2, h2), Image.BILINEAR)
                canvas = Image.new('RGB', (w2 * 2, h2), (255, 255, 255))
                canvas.paste(pil_img, (0, 0))
                canvas.paste(pil_img, (w2, 0))

                draw = ImageDraw.Draw(canvas)
                gt_name = class_names[label]
                pred_name = class_names[pred] if pred < len(class_names) else str(pred)
                is_correct = (pred == label)
                draw.text((5, 5), f"GT: {gt_name}", fill='black', font=font)
                color = 'green' if is_correct else 'red'
                tw, th = draw.textsize(f"Pred: {pred_name}", font=font)
                draw.text((w2 * 2 - tw - 5, 5), f"Pred: {pred_name}", fill=color, font=font)

                frames.append(canvas)

            if all(count >= max_samples_per_class for count in class_sample_count.values()):
                break

    # Generate Confusion Matrix
    cm = confusion_matrix(all_labels, all_preds)
    fig, ax = plt.subplots(figsize=(6, 6))
    sns.heatmap(cm, annot=True, fmt='d', xticklabels=class_names, yticklabels=class_names, cmap="Blues", ax=ax)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    # Save CM as PIL Image
    fig.tight_layout()
    cm_path = "temp_cm.png"
    plt.savefig(cm_path)
    plt.close(fig)

    cm_img = Image.open(cm_path).convert("RGB").resize((frames[0].width, frames[0].height))
    frames.append(cm_img)

    os.makedirs(os.path.dirname(gif_path), exist_ok=True)
    imageio.mimsave(gif_path, frames, duration=500)
    print(f"✅ Combined GIF with Confusion Matrix saved at: {gif_path}")


if __name__=='__main__':
    # Load model
    model = initialize_model(pretrained=True).to(device)
    ckpt = os.path.join(args.output_dir,'best_model.pth')
    model.load_state_dict(torch.load(ckpt, map_location=device))

    # Generate GIFs for each of the three classes
    # class_names = ['happy','neutral','sad']
    # for idx, name in enumerate(class_names):
    #     gif_file = os.path.join(args.output_dir, f"{name}_faces.gif")
    #     inference_and_create_gif_combined(model, val_loader, device, idx, name, gif_file)
    class_names = ['happy', 'neutral', 'sad']
    gif_file = os.path.join(args.output_dir, "combined_faces_with_cm.gif")
    inference_and_create_gif_combined(model, val_loader, device, class_names, gif_file)
#==========================================================================================

# import os
# import random
# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import models, transforms
# from torch.utils.data import DataLoader, random_split
# import argparse

# from dataset import CustomBinaryDataset
# from vgg_face_model import Vgg_face_dag

# # -----------------------
# # ARGPARSE + MODEL SETUP
# # -----------------------
# parser = argparse.ArgumentParser(description='3‑Class Inference & Error‑Plot Script')
# parser.add_argument('--data_dir',      type=str, required=True)
# parser.add_argument('--output_dir',    type=str, default='output')
# parser.add_argument('--mode',          type=str, choices=['vgg_face','resnet18'], default='resnet18')
# parser.add_argument('--checkpoint',    type=str, required=True, help='Path to best_model.pth')
# parser.add_argument('--batch_size',    type=int, default=32)
# parser.add_argument('--device',        type=str, choices=['cuda','cpu'], default='cuda')
# parser.add_argument('--samples_per_cls', type=int, default=20,
#                     help='How many random examples to sample per class before filtering mistakes')
# args = parser.parse_args()

# device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

# def initialize_model(pretrained=True):
#     if args.mode=='vgg_face':
#         model = Vgg_face_dag()
#         model.fc8 = torch.nn.Linear(4096, 3)
#     else:
#         model = models.resnet18(pretrained=pretrained)
#         model.fc = torch.nn.Linear(model.fc.in_features, 3)
#     return model.to(device)

# # -----------------------
# # DATASET & VAL SPLIT
# # -----------------------
# transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
# ])
# full_ds = CustomBinaryDataset(args.data_dir, transform=transform)
# n_val = int(0.2 * len(full_ds))
# _, val_ds = random_split(full_ds, [len(full_ds)-n_val, n_val], generator=torch.Generator().manual_seed(42))

# # -----------------------
# # LOAD MODEL
# # -----------------------
# model = initialize_model(pretrained=True)
# model.load_state_dict(torch.load(args.checkpoint, map_location=device))
# model.eval()

# # -----------------------
# # UTILS: PLOT MISTAKES
# # -----------------------
# def plot_wrong_preds_per_class(dataset, model, device, class_names, samples_per_cls, out_path):
#     # 1) collect indices per class
#     idxs_per_cls = {i: [] for i in range(len(class_names))}
#     for idx in range(len(dataset)):
#         _, lbl = dataset[idx]
#         idxs_per_cls[lbl].append(idx)

#     # 2) for each class, randomly sample, run inference, keep only wrong
#     wrong_images = {i: [] for i in range(len(class_names))}
#     for cls_idx, indices in idxs_per_cls.items():
#         # shuffle & take a superset to ensure enough wrongs
#         random.shuffle(indices)
#         for ds_idx in indices[:samples_per_cls]:
#             img, lbl = dataset[ds_idx]
#             inp = img.unsqueeze(0).to(device)
#             with torch.no_grad():
#                 out = model(inp)
#                 pred = torch.argmax(out, dim=1).item()

#             if pred != cls_idx:
#                 # de-normalize for plotting
#                 inv_mean = torch.tensor([0.485,0.456,0.406]).view(3,1,1)
#                 inv_std  = torch.tensor([0.229,0.224,0.225]).view(3,1,1)
#                 img_denorm = (img * inv_std + inv_mean).clamp(0,1).permute(1,2,0).numpy()
#                 wrong_images[cls_idx].append((img_denorm, pred))
#             if len(wrong_images[cls_idx]) >= 5:  # up to 5 wrongs per class
#                 break

#     # 3) plot grid
#     n_classes = len(class_names)
#     n_cols    = max(len(wrong_images[c]) for c in wrong_images)
#     fig, axs = plt.subplots(n_classes, n_cols, figsize=(n_cols*3, n_classes*3))
#     if n_classes==1: axs = np.expand_dims(axs,0)
#     for row, cls_idx in enumerate(range(n_classes)):
#         for col in range(n_cols):
#             ax = axs[row, col]
#             ax.axis('off')
#             if col < len(wrong_images[cls_idx]):
#                 img_np, pred = wrong_images[cls_idx][col]
#                 ax.imshow(img_np)
#                 title = f"GT: {class_names[cls_idx]}\nPred: {class_names[pred]}"
#                 ax.set_title(title, color='red')
#             else:
#                 ax.set_visible(False)

#     plt.tight_layout()
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.savefig(out_path)
#     print(f"Saved mistake‑plot at: {out_path}")

# # -----------------------
# # MAIN: MAKE & SAVE PLOT
# # -----------------------
# if __name__=='__main__':
#     class_names = ['happy','neutral','sad']
#     out_file = os.path.join(args.output_dir, 'wrong_predictions.png')
#     plot_wrong_preds_per_class(
#         dataset=val_ds,
#         model=model,
#         device=device,
#         class_names=class_names,
#         samples_per_cls=args.samples_per_cls,
#         out_path=out_file
#     )

# import os
# import torch
# from PIL import Image, ImageDraw, ImageFont
# import imageio

# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torchvision import models, transforms, datasets
# from torch.utils.data import DataLoader, random_split
# import argparse
# import time
# import os
# import numpy as np
# import wandb
# from sklearn.metrics import classification_report
# import warnings
# warnings.filterwarnings("ignore")
# from dataset import CustomBinaryDataset
# from vgg_face_model import Vgg_face_dag

# parser = argparse.ArgumentParser(description='ResNet18 Training Script')
# parser.add_argument('--data_dir', type=str, required=True,
#                     help='Path to dataset directory')
# parser.add_argument('--initialization', type=str, required=True,
#                     choices=['pretrained', 'random'],
#                     help='Model initialization: pretrained or random')
# parser.add_argument('--epochs', type=int, default=10,
#                     help='Number of training epochs')
# parser.add_argument('--batch_size', type=int, default=32,
#                     help='Batch size for training')
# parser.add_argument('--lr', type=float, default=0.001,
#                     help='Learning rate')
# parser.add_argument('--output_dir', type=str, default='output',
#                     help='Output directory for models')
# parser.add_argument('--split_size', type=float, default=0.1)
# parser.add_argument('--device', type=str, choices=['cuda', 'cpu'], default='cuda',
#                     help='Device to use for training (cuda/cpu)')
# parser.add_argument('--mode', default = "vgg_face", type= str,
#                     help = "vgg face or resnet18")
# args = parser.parse_args()

# def initialize_model(pretrained=False):
    
#     if args.mode == "vgg_face":
#         model  = Vgg_face_dag()
#         # model.fc8 = nn.Linear(4096, 2)  # Adjust based on actual architecture
#         model.fc8 = nn.Sequential(
#             nn.Dropout(0.5),
#             nn.Linear(4096, 512),
#             nn.ReLU(),
#             nn.Linear(512, 2)
#         )
#     else:
#         model = models.resnet18(pretrained=pretrained)
#         num_ftrs = model.fc.in_features
#         model.fc = nn.Linear(num_ftrs, 2)

#     return model


# device = torch.device(args.device if torch.cuda.is_available() and args.device == 'cuda' else 'cpu')
# if args.device == 'cuda' and not torch.cuda.is_available():
#     print("Warning: CUDA not available, using CPU instead")

# # model = initialize_model(args.initialization == 'pretrained').to(device)


# # Data transformations
# train_transform = transforms.Compose([
#     transforms.RandomResizedCrop(224),
#     transforms.RandomHorizontalFlip(),
#     transforms.RandomRotation(10),
#     transforms.ColorJitter(),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# test_transform = transforms.Compose([
#     transforms.Resize((224,224)),
#     # transforms.CenterCrop(224),
#     transforms.ToTensor(),
#     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# ])

# # Dataset and dataloaders
# dataset_root = "/home/akash/ws/personal/SMAI_assignments/assignment_2/dataset"
# dataset = CustomBinaryDataset(dataset_root, transform=test_transform)

# seed = 42
# generator = torch.Generator().manual_seed(seed)
# train_size = int(0.8 * len(dataset))
# val_size = len(dataset) - train_size
# train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=generator)

# train_loader = DataLoader(
#     train_dataset,
#     batch_size=args.batch_size,
#     shuffle=True,
#     num_workers=4,
#     pin_memory=True if device.type == 'cuda' else False
# )

# val_loader = DataLoader(
#     val_dataset,
#     batch_size=args.batch_size,
#     shuffle=False,
#     num_workers=4,
#     pin_memory=True if device.type == 'cuda' else False
# )


# def inference_and_create_gif(model, loader, device, class_names, gif_path, max_samples=20, scale=3, fps=10):
#     model.eval()
#     frames = []
#     count = 0

#     # Load a larger TrueType font or default
#     try:
#         font = ImageFont.truetype("DejaVuSans-Bold.ttf", size=32)
#     except IOError:
#         font = ImageFont.load_default()

#     duration = 1.0 / fps  # Correct duration in seconds per frame

#     with torch.no_grad():
#         for inputs, labels in loader:
#             inputs = inputs.to(device)
#             labels = labels.to(device)
#             outputs = model(inputs)
#             _, preds = torch.max(outputs, 1)
#             print(">>>>>", outputs)
#             for i in range(inputs.size(0)):
#                 # Prepare image
#                 img = inputs[i].cpu()
#                 mean = torch.tensor([0.485, 0.456, 0.406]).view(3,1,1)
#                 std = torch.tensor([0.229, 0.224, 0.225]).view(3,1,1)
#                 img = img * std + mean
#                 img_np = (img.clamp(0,1).permute(1,2,0).numpy() * 255).astype('uint8')
#                 pil_img = Image.fromarray(img_np)

#                 # Scale up
#                 w, h = pil_img.size
#                 pil_img = pil_img.resize((w * scale, h * scale), Image.BILINEAR)

#                 draw = ImageDraw.Draw(pil_img)
#                 # Get true and predicted labels
#                 true_label = labels[i].item()
#                 pred_label = preds[i].item()
#                 correct = (pred_label == true_label)

#                 # Get class names
#                 gt_class = class_names[true_label]
#                 pred_class = class_names[pred_label]

#                 # Build texts
#                 gt_text = f"GT: {gt_class}"
#                 pred_text = f"Pred: {pred_class}"
#                 color = 'green' if correct else 'red'

#                 # Compute positions for top-right corner text
#                 padding = 10
#                 # Measure text sizes
#                 tw_gt, th_gt = draw.textsize(gt_text, font=font)
#                 tw_pr, th_pr = draw.textsize(pred_text, font=font)
#                 # Position GT at (W - tw_gt - pad, pad)
#                 x_gt = pil_img.width - tw_gt - padding
#                 y_gt = padding
#                 # Position Pred below GT
#                 x_pr = pil_img.width - tw_pr - padding
#                 y_pr = y_gt + th_gt + 5

#                 draw.text((x_gt, y_gt), gt_text, fill='black', font=font)
#                 draw.text((x_pr, y_pr), pred_text, fill=color, font=font)

#                 frames.append(pil_img)
#                 count += 1
#                 if count >= max_samples:
#                     break
#             if count >= max_samples:
#                 break

#     os.makedirs(os.path.dirname(gif_path), exist_ok=True)
#     # Save GIF with corrected duration
#     imageio.mimsave(gif_path, frames, duration=500)
#     print(f"Saved GIF to {gif_path} ({fps} FPS)")


# if __name__ == '__main__':
#     # Load model
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model_path = os.path.join(args.output_dir, 'best_model.pth')
#     model = initialize_model(pretrained=False).to(device)
#     model.load_state_dict(torch.load(model_path, map_location=device))
#     model = initialize_model(args.initialization == 'pretrained').to(device)

#     # Count trainable parameters
#     num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
#     print(f"Number of trainable parameters: {num_params}")

#     # Create GIFs for each class
#     # inference_and_create_gif(
#     #     model=model,
#     #     loader=val_loader,
#     #     device=device,
#     #     class_idx=1,
#     #     class_name='other',
#     #     gif_path=os.path.join(args.output_dir, 'other_faces.gif'),
#     #     max_samples=20
#     # )

#     # inference_and_create_gif(
#     #     model=model,
#     #     loader=val_loader,
#     #     device=device,
#     #     class_idx=0,
#     #     class_name='own',
#     #     gif_path=os.path.join(args.output_dir, 'own_faces.gif'),
#     #     max_samples=20
#     # )

#     class_names = ['own', 'other', ]  # Your class names
#     inference_and_create_gif(
#     model, val_loader, device, 
#     class_names, os.path.join(args.output_dir, 'all_faces.gif'), 
#     max_samples=50, fps=2
# )


import argparse
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader,random_split
from torchvision import transforms, models
from PIL import Image, ImageDraw, ImageFont
import imageio
import numpy as np

from dataset import CustomBinaryDataset
from vgg_face_model import Vgg_face_dag


def initialize_model(mode, pretrained=True, device='cpu'):
    """
    Initialize the model (VGG-Face or ResNet50), load weights if provided, and move to device.
    """
    if mode == "vgg_face":
        model = Vgg_face_dag()
        # Replace the final classification layer
        model.fc8 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 512),
            nn.ReLU(),
            nn.Linear(512, 2)
        )
    else:
        model = models.resnet18(pretrained=pretrained)
        num_ftrs = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(num_ftrs, 256),
            nn.ReLU(),
            nn.Linear(256, 2)
        )

   
    return model



def unnormalize(tensor, mean, std):
    """
    Unnormalize a Torch tensor image and convert to numpy uint8.
    """
    img = tensor.cpu().clone()
    for t, m, s in zip(img, mean, std):
        t.mul_(s).add_(m)
    np_img = np.clip(img.numpy().transpose(1, 2, 0) * 255.0, 0, 255).astype(np.uint8)
    return Image.fromarray(np_img)


def overlay_text(image, gt_label, pred_label):
    """
    Draws GT and prediction labels onto the PIL image.
    """
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default()
    text = f"GT: {gt_label}  Pred: {pred_label}"
    draw.text((20, 20), text, font=font, fill=(255, 0, 0))
    return image


def main():
    parser = argparse.ArgumentParser(description='Inference and GIF creation')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument("--output_dir", type=str)
    parser.add_argument('--mode', choices=['vgg_face', 'resnet18'], default='vgg_face',
                        help='Which model architecture to use')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for inference')
    
    parser.add_argument('--device', choices=['cpu','cuda'], default='cuda',
                        help='Device for inference')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')

    # Define test transforms (must match validation preprocessing)
    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])

    

    test_transform = transforms.Compose([
        transforms.Resize((224,224)),
        # transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    # Dataset and dataloaders
    dataset_root = "/home/akash/ws/personal/SMAI_assignments/assignment_2/dataset/face_test_set"
    dataset = CustomBinaryDataset(dataset_root, transform=test_transform)

 

    val_loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True if device.type == 'cuda' else False
    )

    # Initialize and load model
    model = initialize_model(
        mode=args.mode,
        pretrained=False,
        device=device
    ).to(device)

    model_path = os.path.join(args.output_dir, "best_model.pth")
    model.load_state_dict(torch.load(model_path, map_location=device))

    # Mean and std for un-normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    frames = []
    label_map = {0: 'own', 1: 'other'}

    # Inference loop
    correct = 0
    total = 0
    model.eval()
    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            preds = torch.argmax(outputs, dim=1)
            correct += (preds.cpu() == labels).sum().item()
            total += labels.size(0)

            for img_tensor, gt, pred in zip(inputs, labels, preds):
                pil_img = unnormalize(img_tensor, mean, std)
                annotated = overlay_text(pil_img, label_map[int(gt)], label_map[int(pred)])
                frames.append(annotated)
                # update counters
        accuracy = correct / total if total > 0 else 0
        print(f'Inference Accuracy: {accuracy * 100:.2f}% ({correct}/{total})')

    # Save as GIF
    imageio.mimsave(os.path.join(args.output_dir, "all_faces.gif"), frames, duration=0.5)
    print(f"Saved GIF to {os.path.join(args.output_dir, 'all_faces.gif')}")


if __name__ == '__main__':
    main()

import os
import torch
import torchvision.models as models
from torchvision.models import ResNet50_Weights
from torchvision import transforms
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import datetime
import json
from pytorch_grad_cam import GradCAM, AblationCAM, ScoreCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.image import show_cam_on_image

# Base path
base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'

image_files = [
    'n02098286_West_Highland_white_terrier.JPEG',
    'n02018207_American_coot.JPEG',
    'n04037443_racer.JPEG',
    'n02007558_flamingo.JPEG',
    'n01608432_kite.JPEG',
    'n01443537_goldfish.JPEG',
    'n01491361_tiger_shark.JPEG',
    'n01616318_vulture.JPEG',
    'n01677366_common_iguana.JPEG',
    'n07747607_orange.JPEG'
]


preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model = model.to(device)


cam_methods = {
    'GradCAM': GradCAM,
    'AblationCAM': AblationCAM,
    'ScoreCAM': ScoreCAM
}


with open(os.path.join(base_path, 'data', 'imagenet_classes.txt'), 'r') as f:
    imagenet_classes = [line.strip() for line in f.readlines()]


def generate_cam_heatmaps(model, image_files, save_dir):
    os.makedirs(save_dir, exist_ok=True)
    target_layer = model.layer4[-1]  
    results = []

    for img_file in image_files:
        img_path = os.path.join(base_path, 'data', 'imagenet-sample-images', img_file)
        if not os.path.exists(img_path):
            print(f"Image not found: {img_path}")
            continue


        img = Image.open(img_path).convert('RGB')
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        img_np = np.array(img.resize((224, 224))) / 255.0


        with torch.no_grad():
            output = model(img_tensor)
            pred_idx = output.argmax(dim=1).item()
            pred_class = imagenet_classes[pred_idx]


        for cam_name, cam_class in cam_methods.items():
            cam = cam_class(model=model, target_layers=[target_layer])
            target = [ClassifierOutputTarget(pred_idx)]
            heatmap = cam(input_tensor=img_tensor, targets=target)[0]
            visualization = show_cam_on_image(img_np, heatmap, use_rgb=True)


            plt.figure(figsize=(12, 4))
            plt.subplot(1, 3, 1)
            plt.imshow(img_np)
            plt.title('Original Image')
            plt.axis('off')
            plt.subplot(1, 3, 2)
            plt.imshow(heatmap, cmap='jet')
            plt.title(f'{cam_name} Heatmap')
            plt.axis('off')
            plt.subplot(1, 3, 3)
            plt.imshow(visualization)
            plt.title(f'{cam_name} Overlay')
            plt.axis('off')
            plt.suptitle(f'{img_file} - Predicted: {pred_class}')
            plt.tight_layout()
            save_path = os.path.join(save_dir, f'{cam_name}_{img_file[:-5]}_cam.png')
            plt.savefig(save_path)
            plt.close()

            results.append({
                'image': img_file,
                'cam_method': cam_name,
                'predicted_class': pred_class,
                'cam_image': save_path
            })

    return results


save_dir = os.path.join(base_path, 'scripts')
results = generate_cam_heatmaps(model, image_files, save_dir)


save_path = os.path.join(base_path, 'results', f'gradcam_summary_{datetime.datetime.now().strftime("%y_%m_%d_%H_%M")}')
os.makedirs(save_path, exist_ok=True)
with open(os.path.join(save_path, 'gradcam_summary.json'), 'w') as f:
    json.dump(results, f, indent=2)

print("CAM analysis completed. Results saved in", save_path)
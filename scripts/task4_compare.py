import os
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from torchvision.models import ResNet50_Weights
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from lime import lime_image
from skimage.segmentation import mark_boundaries
import pandas as pd
import datetime
import json

# Base path
base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'
output_dir = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21\results\output-png\Task4-Results'
report_dir = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21\results'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(report_dir, exist_ok=True)

# Image names
image_names = [
    'n02098286_West_Highland_white_terrier',
    'n02018207_American_coot',
    'n04037443_racer',
    'n02007558_flamingo',
    'n01608432_kite',
    'n01443537_goldfish',
    'n01491361_tiger_shark',
    'n01616318_vulture',
    'n01677366_common_iguana',
    'n07747607_orange'
]

# Load ImageNet classes
with open(os.path.join(base_path, 'data', 'imagenet_classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Preprocessing pipeline
preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V2)
model.eval()
model = model.to(device)

# LIME prediction function
def predict_proba(images):
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    images = images.to(device)
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1).cpu().numpy()
    return probs

# LIME parameters
lime_params = {
    "labels": None,
    "top_labels": 1,
    "hide_color": 0,
    "num_features": 100000,
    "num_samples": 1000,
    "batch_size": 10,
    "segmentation_fn": None,
    "distance_metric": "cosine",
    "model_regressor": None,
    "random_seed": None
}

# Initialize LIME explainer
explainer = lime_image.LimeImageExplainer()

# Function to find image file
def find_image_file(base_path, image_name):
    for ext in ['.jpg', '.JPEG', '.jpeg', '.png']:
        img_path = os.path.join(base_path, image_name + ext)
        if os.path.exists(img_path):
            return img_path
    raise FileNotFoundError(f"Image {image_name} not found")

# Function to compute IoU
def compute_iou(mask1, mask2, threshold=0.5):
    mask1_binary = (mask1 > threshold).astype(np.uint8)
    mask2_binary = (mask2 > threshold).astype(np.uint8)
    intersection = np.logical_and(mask1_binary, mask2_binary).sum()
    union = np.logical_or(mask1_binary, mask2_binary).sum()
    if union == 0:
        return 0.0
    return intersection / union

# Process images and generate Grad-CAM and LIME results
results = []
ious = []
for img_name in image_names:
    try:
        img_path = find_image_file(os.path.join(base_path, 'data', 'imagenet-sample-images'), img_name)
    except FileNotFoundError as e:
        print(e)
        continue

    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img.resize((224, 224))) / 255.0
    img_tensor = preprocess(img).unsqueeze(0).to(device)

    # Get model prediction
    with torch.no_grad():
        output = model(img_tensor)
        pred_idx = output.argmax(dim=1).item()
        pred_class = classes[pred_idx]

    # Generate Grad-CAM heatmap
    target_layer = model.layer4[-1]
    cam = GradCAM(model=model, target_layers=[target_layer])
    target = [ClassifierOutputTarget(pred_idx)]
    gradcam_heatmap = cam(input_tensor=img_tensor, targets=target)[0]

    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=predict_proba,
        **lime_params
    )
    _, lime_mask = explanation.get_image_and_mask(
        explanation.top_labels[0],
        positive_only=False,
        num_features=10,
        hide_rest=False
    )

    # Compute IoU
    iou = compute_iou(gradcam_heatmap, lime_mask)

    # Save visualizations
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title('Original Image')
    plt.axis('off')
    plt.subplot(1, 3, 2)
    plt.imshow(gradcam_heatmap, cmap='jet')
    plt.title('Grad-CAM Heatmap')
    plt.axis('off')
    plt.subplot(1, 3, 3)
    plt.imshow(mark_boundaries(img_np, lime_mask))
    plt.title('LIME Explanation')
    plt.axis('off')
    plt.suptitle(f'{img_name} - Predicted: {pred_class} (IoU: {iou:.4f})')
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'comparison_{img_name}.png')
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()

    results.append({
        'image': img_name,
        'predicted_class': pred_class,
        'iou': iou,
        'gradcam_path': save_path,
        'lime_path': save_path
    })
    ious.append(iou)
    print(f'Processed {img_name}, IoU: {iou:.4f}')


# Save results summary
summary_path = os.path.join(report_dir, 'task4_summary.json')
with open(summary_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Visualizations saved to {output_dir}")
print(f"Summary saved to {summary_path}")
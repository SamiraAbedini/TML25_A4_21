import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Base path
base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'

# Output paths
output_dir = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21\results\output-png\Task3-Results\2'
raw_output_dir = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21\results\output-png\Task3-Results\Raw'

# Create output directories
os.makedirs(output_dir, exist_ok=True)
os.makedirs(raw_output_dir, exist_ok=True)

# Load ImageNet classes
with open(os.path.join(base_path, 'data', 'imagenet_classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Prediction function for LIME
def predict_proba(images):
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    images = images.to(torch.device('cpu'))
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1).numpy()
    return probs

# Function to find image file with multiple extensions
def find_image_file(base_path, image_name):
    for ext in ['.jpg', '.JPEG', '.jpeg', '.png']:
        img_path = os.path.join(base_path, image_name + ext)
        if os.path.exists(img_path):
            return img_path
    available_files = os.listdir(base_path)
    raise FileNotFoundError(f"Image {image_name} not found with .jpg, .JPEG, .jpeg, or .png extension. Available files: {available_files}")

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

# Summary dictionary
summary = {}

# Process each image
for img_name in image_names:
    try:
        img_path = find_image_file(os.path.join(base_path, 'data', 'imagenet-sample-images'), img_name)
    except FileNotFoundError as e:
        print(e)
        continue
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    # Generate LIME explanation
    explanation = explainer.explain_instance(
        image=img_np,
        classifier_fn=predict_proba,
        **lime_params
    )
    
    # Get image and mask for visualization
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=10, 
        hide_rest=False
    )
    
    # Generate boundary visualization
    img_boundry = mark_boundaries(temp, mask)
    
    # Save LIME visualization with boundaries
    output_path = os.path.join(output_dir, f'LIME_{img_name}.png')
    plt.figure()
    plt.imshow(img_boundry)
    plt.title(f'LIME: {classes[explanation.top_labels[0]]} ({img_name})')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Save raw LIME mask
    raw_output_path = os.path.join(raw_output_dir, f'LIME_raw_{img_name}.png')
    plt.figure(figsize=(224/100, 224/100), dpi=100)
    plt.imshow(mask, cmap='jet')
    plt.axis('off')
    plt.savefig(raw_output_path, bbox_inches='tight', pad_inches=0)
    plt.close()
    
    # Store results in summary
    summary[img_name] = {
        'image_path': img_path,
        'lime_output': output_path,
        'predicted_class': classes[explanation.top_labels[0]],
        'class_index': int(explanation.top_labels[0])
    }
    
    print(f'Processed {img_name}')

# Save summary
summary_path = os.path.join(base_path, 'results', 'lime_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)

print(f'Saved summary to {summary_path}')
print(f'LIME visualizations saved in {output_dir}')
print(f'Raw LIME masks saved in {raw_output_dir}')
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import torch
from torchvision import models, transforms
from lime import lime_image
from skimage.segmentation import mark_boundaries

# Set base path
base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'

# Load ImageNet class names
with open(os.path.join(base_path, 'data', 'imagenet_classes.txt'), 'r') as f:
    classes = [line.strip() for line in f.readlines()]

# Define image preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Load ResNet50 model
model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
model.eval()

# Function to predict probabilities for LIME
def predict_proba(images):
    images = torch.from_numpy(images.transpose((0, 3, 1, 2))).float()
    images = images.to(torch.device('cpu'))
    with torch.no_grad():
        outputs = model(images)
        probs = torch.nn.functional.softmax(outputs, dim=1).numpy()
    return probs

# Function to find image with correct extension
def find_image_file(base_path, image_name):
    for ext in ['.jpg', '.JPEG', '.jpeg', '.png']:
        img_path = os.path.join(base_path, image_name + ext)
        if os.path.exists(img_path):
            return img_path
    # Log available files for debugging
    available_files = os.listdir(base_path)
    raise FileNotFoundError(f"Image {image_name} not found with .jpg, .JPEG, .jpeg, or .png extension. Available files: {available_files}")

# Image list (corrected based on Task 2 outputs)
image_dir = os.path.join(base_path, 'data', 'imagenet-sample-images')
image_names = [
    'n02098286_West_Highland_white_terrier',
    'n02018207_American_coot',  # Corrected from n01592084
    'n04037443_racer',  # Corrected from n01751748
    'n02007558_flamingo',  # Corrected from n02025239
    'n01608432_kite',  # Corrected from n01514668
    'n01443537_goldfish',
    'n01491361_tiger_shark',
    'n01616318_vulture',  # Corrected from n01693334
    'n01677366_common_iguana',  # Corrected from n01729322
    'n07747607_orange'
]

# Output directories
output_dir = os.path.join(base_path, 'scripts')
results_dir = os.path.join(base_path, 'results')
os.makedirs(output_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# LIME explainer
explainer = lime_image.LimeImageExplainer()

# Summary dictionary
summary = {}

# Process each image
for img_name in image_names:
    # Find image file
    try:
        img_path = find_image_file(image_dir, img_name)
    except FileNotFoundError as e:
        print(e)
        continue
    
    # Load and preprocess image
    img = Image.open(img_path).convert('RGB')
    img_t = transform(img)
    img_np = np.array(img.resize((224, 224))) / 255.0
    
    # Generate LIME explanation
    explanation = explainer.explain_instance(
        img_np, 
        predict_proba, 
        top_labels=1, 
        hide_color=0, 
        num_samples=1000
    )
    
    # Get image and mask for top predicted class
    temp, mask = explanation.get_image_and_mask(
        explanation.top_labels[0], 
        positive_only=False, 
        num_features=10, 
        hide_rest=False
    )
    
    # Create visualization
    img_boundry = mark_boundaries(temp, mask)
    
    # Save visualization
    output_path = os.path.join(output_dir, f'LIME_{img_name}.png')
    plt.figure()
    plt.imshow(img_boundry)
    plt.title(f'LIME: {classes[explanation.top_labels[0]]} ({img_name})')
    plt.axis('off')
    plt.savefig(output_path, bbox_inches='tight')
    plt.close()
    
    # Update summary
    summary[img_name] = {
        'image_path': img_path,
        'lime_output': output_path,
        'predicted_class': classes[explanation.top_labels[0]],
        'class_index': int(explanation.top_labels[0])
    }
    
    print(f'Processed {img_name}')

# Save summary
summary_path = os.path.join(results_dir, 'lime_summary.json')
with open(summary_path, 'w') as f:
    json.dump(summary, f, indent=4)

print(f'Saved summary to {summary_path}')

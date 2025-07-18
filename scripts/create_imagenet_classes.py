import os
from torchvision.models import ResNet50_Weights

# Base path
base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'

# Get ImageNet class names from ResNet50 weights
weights = ResNet50_Weights.IMAGENET1K_V2
classes = weights.meta['categories']

# Save to file
output_path = os.path.join(base_path, 'data', 'imagenet_classes.txt')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
with open(output_path, 'w') as f:
    for cls in classes:
        f.write(cls + '\n')

print(f"Saved {output_path} with {len(classes)} classes")
import json
import matplotlib.pyplot as plt
import os

base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'
json_path = os.path.join(base_path, 'results', 'task4_summary.json')
output_dir = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21\results\output-png\Task4-Results'

os.makedirs(output_dir, exist_ok=True)

with open(json_path, 'r') as f:
    data = json.load(f)

image_names = [entry['image'] for entry in data]
ious = [entry['iou'] for entry in data]

short_names = [name.split('_', 1)[1].replace('_', ' ').title() for name in image_names]

plt.figure(figsize=(12, 6))
bars = plt.bar(short_names, ious, color='skyblue', edgecolor='navy')

for bar, iou in zip(bars, ious):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{iou:.4f}', 
             ha='center', va='bottom', fontsize=10)

plt.xlabel('Image', fontsize=12)
plt.ylabel('IoU (Grad-CAM vs LIME)', fontsize=12)
plt.title('IoU Comparison Between Grad-CAM and LIME for ImageNet Images', fontsize=14)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, max(ious) + 0.1)  
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()

output_path = os.path.join(output_dir, 'iou_comparison_bar_plot.png')
plt.savefig(output_path, bbox_inches='tight', dpi=300)
plt.close()

print(f"Bar plot saved to {output_path}")
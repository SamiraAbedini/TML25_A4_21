import os
import torch
import torchvision.models as models
from torchvision.models import ResNet18_Weights
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import clip
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter
import numpy as np
import datetime
import json
import sys
from tqdm import tqdm


base_path = r'F:\A-Saarland-University-Courses\TML\TML25_A4_21'
sys.path.append(os.path.join(base_path, 'clip_dissect'))
try:
    import utils
    import similarity
except ImportError:
    print("Error: CLIP-dissect modules (utils, similarity) not found in 'clip_dissect'. Ensure they are present.")
    sys.exit(1)


data_dir = os.path.join(base_path, 'data')
os.makedirs(data_dir, exist_ok=True)


class ImageNetSampleDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_files = [f for f in os.listdir(root_dir) if f.endswith('.JPEG')]
        if not self.image_files:
            raise FileNotFoundError(f"No .JPEG files found in {root_dir}")
        self.classes = sorted(set(f.split('_')[0] for f in self.image_files))
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')
        class_id = self.image_files[idx].split('_')[0]
        label = self.class_to_idx[class_id]
        if self.transform:
            image = self.transform(image)
        return image, label, img_path


preprocess = transforms.Compose([
    transforms.Resize(256, antialias=True),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


concept_set_path = os.path.join(base_path, 'data', '20k.txt')
image_dir = os.path.join(base_path, 'data', 'imagenet-sample-images')
if not os.path.exists(concept_set_path):
    try:
        concepts = [f.split('_', 1)[1].replace('.JPEG', '').replace('_', ' ') for f in os.listdir(image_dir) if f.endswith('.JPEG')]
        concepts = sorted(set(concepts))
        with open(concept_set_path, 'w') as f:
            f.write('\n'.join(concepts))
        print(f"Created concept set with {len(concepts)} concepts: {concept_set_path}")
    except FileNotFoundError as e:
        print(f"Error creating concept set: {e}")
        sys.exit(1)


try:
    dataset = ImageNetSampleDataset(root_dir=image_dir, transform=preprocess)
    dataloader = DataLoader(dataset, batch_size=32, shuffle=False)
except FileNotFoundError as e:
    print(f"Error loading dataset: {e}")
    sys.exit(1)


device = 'cuda' if torch.cuda.is_available() else 'cpu'


model_imagenet = models.resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)
model_imagenet.eval()
model_imagenet = model_imagenet.to(device)


model_places365 = None
try:
    model_places365 = models.resnet18()
    state_dict = torch.load(os.path.join(base_path, 'data', 'resnet18_places365.pth.tar'), map_location=device)
    print("Places365 state_dict keys:", list(state_dict.keys())[:10])
    if 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    if any(k.startswith('module.') for k in state_dict.keys()):
        state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
    if state_dict['fc.weight'].shape[0] != 1000:
        print("Adjusting fc layer for Places365 (365 classes)")
        model_places365.fc = torch.nn.Linear(model_places365.fc.in_features, 365)
    model_places365.load_state_dict(state_dict, strict=False)
    model_places365.eval()
    model_places365 = model_places365.to(device)
except Exception as e:
    print(f"Error loading Places365 model: {e}. Continuing with ImageNet model only.")


try:
    clip_model, clip_preprocess = clip.load('ViT-B/16', device=device)
except Exception as e:
    print(f"Error loading CLIP model: {e}")
    sys.exit(1)


def run_network_dissection(model, model_name, target_layers, dataloader, concept_set, activation_dir, result_dir):
    args = {
        'clip_model': 'ViT-B/16',
        'target_model': model_name,
        'target_layers': target_layers,
        'd_probe': 'custom_imagenet',
        'concept_set': os.path.basename(concept_set),   
        'batch_size': 32,
        'device': device,
        'activation_dir': activation_dir,
        'result_dir': result_dir,
        'pool_mode': 'avg',
        'similarity_fn': 'soft_wpmi'
    }


    os.makedirs(args['activation_dir'], exist_ok=True)
    os.makedirs(args['result_dir'], exist_ok=True)


    def save_activations(clip_name, target_name, target_layers, d_probe, concept_set, batch_size, device, pool_mode, save_dir, dataset):

        with open(concept_set, 'r') as f:
            words = f.read().split('\n')
        clip_texts = clip.tokenize(words).to(device)
        with torch.no_grad():
            text_features = clip_model.encode_text(clip_texts)


        activations = {}
        def hook_fn(module, input, output):
            if pool_mode == 'avg':
                activations[module] = output.mean(dim=[2, 3])
            else:
                activations[module] = output.max(dim=2)[0].max(dim=2)[0]

        handles = []
        for layer_name in target_layers:
            layer = dict(model.named_modules())[layer_name]
            handle = layer.register_forward_hook(hook_fn)
            handles.append(handle)


        target_acts = {layer: [] for layer in target_layers}
        clip_acts = []
        image_paths = []
        to_pil = transforms.ToPILImage()
        with torch.no_grad():
            for batch in tqdm(dataset, desc=f"Processing {target_name}"):
                images, _, paths = batch
                images = images.to(device)
                

                model(images)
                for layer_name in target_layers:
                    layer = dict(model.named_modules())[layer_name]
                    target_acts[layer_name].append(activations[layer].cpu())
                

                clip_images = []
                for img in images:
                    img = to_pil(img.cpu()) 
                    clip_images.append(clip_preprocess(img))
                clip_images = torch.stack(clip_images).to(device)
                clip_features = clip_model.encode_image(clip_images)
                clip_acts.append(clip_features.cpu())
                image_paths.extend(paths)


        for layer_name in target_layers:
            save_names = utils.get_save_names(
                clip_name=clip_name,
                target_name=target_name,
                target_layer=layer_name,
                d_probe=d_probe,
                concept_set=os.path.basename(concept_set), 
                pool_mode=pool_mode,
                save_dir=save_dir
            )
            target_save_name, clip_save_name, text_save_name = save_names
            print(f"Saving activations: {target_save_name}, {clip_save_name}, {text_save_name}")
            torch.save(torch.cat(target_acts[layer_name], dim=0), target_save_name)
            torch.save(torch.cat(clip_acts, dim=0), clip_save_name)
            torch.save(text_features.cpu(), text_save_name)


        for handle in handles:
            handle.remove()

        return image_paths


    image_paths = save_activations(
        clip_name=args['clip_model'],
        target_name=args['target_model'],
        target_layers=args['target_layers'],
        d_probe=args['d_probe'],
        concept_set=concept_set, 
        batch_size=args['batch_size'],
        device=args['device'],
        pool_mode=args['pool_mode'],
        save_dir=args['activation_dir'],
        dataset=dataloader
    )


    similarity_fn = eval("similarity.{}".format(args['similarity_fn']))
    outputs = {"layer": [], "unit": [], "description": [], "similarity": [], "top_images": []}
    with open(concept_set, 'r') as f:
        words = f.read().split('\n')

    for target_layer in args['target_layers']:
        save_names = utils.get_save_names(
            clip_name=args['clip_model'],
            target_name=args['target_model'],
            target_layer=target_layer,
            d_probe=args['d_probe'],
            concept_set=args['concept_set'],
            pool_mode=args['pool_mode'],
            save_dir=args['activation_dir']
        )
        target_save_name, clip_save_name, text_save_name = save_names
        print(f"Loading activations for {target_layer}: {target_save_name}, {clip_save_name}, {text_save_name}")
        try:
            result = utils.get_similarity_from_activations(
                target_save_name, clip_save_name, text_save_name, similarity_fn,
                return_target_feats=True, device=args['device']
            )

            if isinstance(result, tuple):
                similarities, target_features = result
            else:
                similarities = result
                target_features = torch.load(target_save_name)

            print(f"Similarities shape: {similarities.shape}")
            vals, ids = torch.max(similarities, dim=1)
            descriptions = [words[int(idx)] for idx in ids]


            top_images = []
            for unit in range(len(vals)):
                unit_activations = target_features[:, unit]
                top_indices = unit_activations.argsort(descending=True)[:5]
                top_images.append([image_paths[int(i)] for i in top_indices])

            outputs["unit"].extend([i for i in range(len(vals))])
            outputs["layer"].extend([target_layer] * len(vals))
            outputs["description"].extend(descriptions)
            outputs["similarity"].extend(vals.cpu().numpy())
            outputs["top_images"].extend(top_images)

            del similarities, target_features
            torch.cuda.empty_cache()
        except Exception as e:
            print(f"Error processing layer {target_layer}: {e}")
            continue


    df = pd.DataFrame(outputs)
    save_path = os.path.join(args['result_dir'], f"{args['target_model']}_{datetime.datetime.now().strftime('%y_%m_%d_%H_%M')}")
    os.makedirs(save_path, exist_ok=True)
    df.to_csv(os.path.join(save_path, "descriptions.csv"), index=False)
    with open(os.path.join(save_path, "args.txt"), 'w') as f:
        json.dump(args, f, indent=2)
    return df


target_layers = ['layer2', 'layer3', 'layer4']
results = {}
models_to_process = [('imagenet', model_imagenet)]
if model_places365 is not None:
    models_to_process.append(('places', model_places365))

for model_name, model in models_to_process:
    try:
        df = run_network_dissection(
            model=model,
            model_name=f"resnet18_{model_name}",
            target_layers=target_layers,
            dataloader=dataloader,
            concept_set=concept_set_path,
            activation_dir=os.path.join(base_path, 'saved_activations'),
            result_dir=os.path.join(base_path, 'results')
        )
        results[model_name] = df
    except Exception as e:
        print(f"Error running dissection for {model_name}: {e}")
        continue


concept_counts = {}
for model_name, df in results.items():
    concept_counts[model_name] = {}
    for layer in target_layers:
        layer_df = df[df['layer'] == layer]
        concept_counts[model_name][layer] = Counter(layer_df['description'])


imagenet_concepts = set(concept_counts.get('imagenet', {}).get('layer4', {}).keys())
places365_concepts = set(concept_counts.get('places', {}).get('layer4', {}).keys())
shared_concepts = imagenet_concepts.intersection(places365_concepts)
unique_imagenet = imagenet_concepts - places365_concepts
unique_places365 = places365_concepts - imagenet_concepts


print(f"Shared concepts (layer4): {len(shared_concepts)}")
print(f"Unique to ImageNet (layer4): {len(unique_imagenet)}")
print(f"Unique to Places365 (layer4): {len(unique_places365)}")


os.makedirs(os.path.join(base_path, 'scripts'), exist_ok=True)
for model_name, layer_counts in concept_counts.items():
    for layer, counts in layer_counts.items():
        top_concepts = counts.most_common(10)
        if not top_concepts:   
            print(f"No concepts found for {model_name} {layer}, skipping histogram")
            continue
        concepts, counts = zip(*top_concepts)
        plt.figure(figsize=(10, 6))
        plt.bar(concepts, counts)
        plt.xticks(rotation=45, ha='right')
        plt.xlabel('Concepts')
        plt.ylabel('Number of Neurons')
        plt.title(f'{model_name.capitalize()} - {layer} Top Concepts')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, 'scripts', f'{model_name}_{layer}_histogram.png'))
        plt.close()


for model_name, df in results.items():
    layer_df = df[df['layer'] == 'layer4']
    dog_neuron = layer_df[layer_df['description'] == 'dog'].head(1)
    if not dog_neuron.empty:
        unit = dog_neuron['unit'].iloc[0]
        top_images = dog_neuron['top_images'].iloc[0][:3]
        plt.figure(figsize=(12, 4))
        for i, img_path in enumerate(top_images):
            img = Image.open(img_path)
            plt.subplot(1, 3, i+1)
            plt.imshow(img)
            plt.title(f'Dog Image {i+1}')
            plt.axis('off')
        plt.suptitle(f'{model_name.capitalize()} - Layer4 Neuron {unit}')
        plt.tight_layout()
        plt.savefig(os.path.join(base_path, 'scripts', f'{model_name}_layer4_dog_images.png'))
        plt.close()


plt.figure(figsize=(8, 6))
plt.bar(['Shared', 'ImageNet Only', 'Places365 Only'], 
        [len(shared_concepts), len(unique_imagenet), len(unique_places365)],
        color=['#4CAF50', '#2196F3', '#FF9800'])
plt.ylabel('Number of Concepts')
plt.title('Concept Overlap between Models (Layer4)')
plt.tight_layout()
plt.savefig(os.path.join(base_path, 'scripts', 'concept_overlap.png'))
plt.close()


for model_name in concept_counts:
    diversity = {layer: len(counts) for layer, counts in concept_counts[model_name].items()}
    print(f"{model_name.capitalize()} concept diversity: {diversity}")
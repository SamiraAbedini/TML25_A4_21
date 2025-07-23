# TML25_A4_21 

## Explainability 

This repository contains code for analyzing neural network interpretability using Network Dissection, Grad-CAM, Ablation-CAM, Score-CAM and LIME on ResNet models trained on ImageNet and Places365 datasets. Below is an overview of the project structure and instructions to run the code.

## Directory Structure

- **data/**: Contains input data and preprocessed files.
  - `imagenet-sample-images/`: Sample ImageNet images for analysis.
  - `imagenet_classes.txt`: List of ImageNet class names.
  - `20k.txt`: Concept set for Network Dissection.
- **scripts/**: Python scripts for the tasks.
  - `create_imagenet_classes.py`: Generates `imagenet_classes.txt` from ResNet50 weights.
  - `submit_lime_config.py`: Configures and submits LIME parameters for evaluation on the score board.
  - `task1_network_dissection.py`: Performs Network Dissection on ResNet18 (ImageNet and Places365).
  - `task2_gradcam.py`: Generates Grad-CAM, AblationCAM, and ScoreCAM visualizations.
  - `task3_lime.py`: Applies LIME to explain ResNet50 predictions.
  - `task4_compare.py`: Computes the IoU between Lime and Grad-CAM results and creates figures.
  - `task4-plot.py`: Creates IoU comparison bar plot.
- **results/**: Stores output files (e.g., CSVs, JSONs, and visualization directories).
  - `gradcam_summary_*/`: Grad-CAM results and visualizations.
  - `resnet18_imagenet_*/` and `resnet18_places_*/`: Here you can find the concepts learned by each neuron in the given layers.
  - `lime_summary.json`: LIME analysis summary.
  - `output-png` : This folder includes the results of visualization of different CAM methods, LIME and network dissection.
- **saved_activations/**: Stores activation data from Network Dissection.
- **content/**: Used for LIME parameter storage (`explain_params.pkl`). The file in the directory is submitted to the score board.

## Prerequisites

- **Dependencies**: Install required packages using:
  ```bash
  pip install -r requirements.txt
  ```
- **Places365 Model**: Download `resnet18_places365.pth.tar` from the [Places365 repository](http://places2.csail.mit.edu/models_places365/resnet18_places365.pth.tar) and place it in `data/`.
- **CLIP-dissect**: In order to run `task1_network_dissection.py` make sure to clone the following repository using:
```bash
  git clone https://github.com/Trustworthy-ML-Lab/CLIP-dissect.git
  ```
  Otherwise you will face dependency issue.

## Setup

1. **Prepare Data**:
   - Place ImageNet sample images in `data/imagenet-sample-images/`.
   - Ensure `resnet18_places365.pth.tar` is in `data/`.

## Running the Code

1. **Generate ImageNet Classes**:
   ```bash
   python scripts/create_imagenet_classes.py
   ```
   - Creates `data/imagenet_classes.txt` with ImageNet class names.

2. **Network Dissection (Task 1)**:
   ```bash
   python scripts/task1_network_dissection.py
   ```
   - Analyzes ResNet18 (ImageNet and Places365) for concept activation.
   - Outputs: CSVs and histograms in `results/resnet18_[imagenet/places]_*/` and `scripts/`.

3. **Grad-CAM (Task 2)**:
   ```bash
   python scripts/task2_gradcam.py
   ```
   - Generates heatmaps using Grad-CAM, AblationCAM, and ScoreCAM for specified images.
   - Outputs: Visualizations in `scripts/` and summary in `results/gradcam_summary_*/`.

4. **LIME (Task 3)**:
   ```bash
   python scripts/task3_lime.py
   ```
   - Applies LIME to explain ResNet50 predictions.
   - Outputs: Visualizations in `scripts/` and `results/lime_summary.json`.

5. **Submit LIME Config**:
   ```bash
   python scripts/submit_lime_config.py
   ```
   - Submits LIME parameters to a remote server (requires internet and valid token).
   - Outputs: Response JSON with evaluation metrics.

## Notes

- **File Paths**: Ensure all paths in scripts match your local setup.
- **Errors**: If images or `clip_dissect` modules are missing, scripts will exit with error messages. Verify the `data/` directory and `clip_dissect/` contents.
- **Output**: Results are timestamped and saved in `results/` with visualizations in `scripts/`.

For further details of methods and achieved results, refer to the scripts or the report document (`Report Assignment Sheet.docx`).
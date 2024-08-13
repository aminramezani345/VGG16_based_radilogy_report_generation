# ViT for Radilogy Report generation from Chest X-ray

This repository contains code for generating medical reports from frontal and lateral chest X-ray images. It utilizes a deep learning model to process the images and generate descriptive reports. In this version, the visual extractor model has been changed from ResNet101 to VGG16, resulting in improved performance.

## Overview

The code uses a Flask web application to accept image uploads, process them using a trained model, and return generated reports. The core model for extracting visual features and generating reports has been updated from ResNet101 to VGG16, which has shown better results in performance evaluations.

## References

The base code for the visual extractor models and other configurations is derived from the [TSGET repository](https://github.com/SKD-HPC/TSGET).

## Installation

To get started, clone this repository and install the required dependencies:

```bash
git clone <repository_url>
cd <repository_directory>
pip install -r requirements.txt

## Configuration

### Command-Line Arguments

The following command-line arguments can be used to configure the model and training process:

- `--image_dir`: Path to the directory containing the images.
- `--ann_path`: Path to the annotation file.
- `--dataset_name`: Name of the dataset.
- `--max_seq_length`: Maximum sequence length of the reports.
- `--threshold`: Cut-off frequency for the words.
- `--num_workers`: Number of workers for the data loader.
- `--batch_size`: Number of samples per batch.
- `--visual_extractor`: The visual extractor model to be used (default: `vgg16`).
- `--visual_extractor_pretrained`: Whether to load the pretrained visual extractor (default: `True`).
- `--d_model`: Dimension of Transformer model (default: `512`).
- `--d_ff`: Dimension of FeedForward Network (default: `512`).
- `--num_heads`: Number of attention heads (default: `8`).
- `--num_layers`: Number of Transformer layers (default: `3`).
- `--dropout`: Dropout rate for Transformer (default: `0.1`).
- `--sample_method`: Method for sampling a report (default: `beam_search`).
- `--beam_size`: Beam size for beam searching (default: `3`).
- `--temperature`: Temperature for sampling (default: `1.0`).
- `--epochs`: Number of training epochs (default: `100`).
- `--save_dir`: Directory to save the models (default: `/checkpoints`).
- `--record_dir`: Directory to save experiment results (default: `records/`).
## Visual Extractor

In this implementation, the visual extractor model has been changed to `vgg16` from the previous `resnet101`. The visual extractor settings can be modified using:

- `--visual_extractor`: Set this argument to `vgg16` to use the VGG16 model.

## Running the Application

To start the Flask web application, execute the following command:

```bash
python app.py


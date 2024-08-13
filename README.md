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



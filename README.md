# User Interface for implement ViT in Radilogy Report generation from Chest X-ray

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
```
## Configuration

## Comman Line Argument

The following command-line arguments can be used to configure the model and training process:

```bash
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
```
## Visual Extractor

In this implementation, the visual extractor model has been changed to `vgg16` from the previous `resnet101`. The visual extractor settings can be modified using:

```bash
- `--visual_extractor`: Set this argument to `vgg16` to use the VGG16 model.
```
## Running the Application

To start the Flask web application, execute the following command:

```bash
python app.py
```
This will launch a web server on http://127.0.0.1:5000. You can access the application through this URL.

## Usage
Open the web application in your browser.
Upload frontal and lateral chest X-ray images.
The application will process the images and return a generated report.
To use the Flask web application described in the README, follow these steps:

1. Clone the Repository
First, clone the repository to your local machine if you haven't done so already:

```bash
git clone https://github.com/your-username/your-repository.git
cd your-repository
```
2. Set Up Your Environment
Ensure you have Python installed. Create and activate a virtual environment:

```bash
python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
```
3. Install Dependencies
Install the required Python packages:
```bash
pip install flask
```
4. Configure the Application
If you need to modify any configuration settings, you can update the command-line arguments in the app.py file or set them directly in your script.

5. Prepare Your Model
Ensure you have the trained model checkpoint available. Place your model checkpoint file in the appropriate directory (e.g., ./checkpoints/current_checkpoint.pth).

6. Run the Flask Application
Start the Flask web server by running:

```bash
python app.py
```
This will start the server on http://127.0.0.1:5000. You should see output indicating the server is running.

7. Access the Web Application
Open a web browser and navigate to http://127.0.0.1:5000. You should see the main page of the application.

8. Use the Application
On the web page, you will find an option to upload frontal and lateral chest X-ray images.
Upload the images you want to process.
The application will generate a report based on the uploaded images.
9. View the Results
After processing, the application will display the generated report, which you can review or use as needed.

10. Shut Down the Server
To stop the server, you can interrupt the process in your terminal by pressing Ctrl+C.

By following these steps, you should be able to run and use the Flask web application effectively. If you encounter any issues, refer to the error messages in the terminal or consult the project's documentation for further troubleshooting.
## License
This project is licensed under the MIT License. See the LICENSE file for details.

## Acknowledgements
Special thanks to the authors of the TSGET repository for their foundational code and models.


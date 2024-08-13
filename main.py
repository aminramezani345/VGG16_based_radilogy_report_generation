
#!pip install flask
from flask import Flask, request, render_template, redirect, url_for
import torch
from torchvision import transforms
from PIL import Image
import os
from werkzeug.utils import secure_filename
import os
import webbrowser
import time
from flask import Flask, render_template, request
from transformers import pipeline, BartTokenizer, BartForConditionalGeneration
import threading
import fitz  # PyMuPDF
import torch
import argparse
import numpy as np
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader
from modules.metrics import compute_scores
from modules.optimizers import build_optimizer, build_lr_scheduler
from modules.trainer import Trainer
from modules.loss import compute_loss
from models.r2gen import R2GenModel
from torchvision import transforms
from PIL import Image
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io

app = Flask(__name__)

def parse_agrs():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='data/iu_xray/images/', help='the path to the directory containing the data.')
    parser.add_argument('--ann_path', type=str, default='data/iu_xray/annotation.json', help='the path to the directory containing the data.')

    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='iu_xray', help='the dataset to be used.')
    parser.add_argument('--max_seq_length', type=int, default=100, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=16, help='the number of samples for a batch')

    # Model settings (for visual extractor)
    parser.add_argument('--visual_extractor', type=str, default='vgg16', help='the visual extractor to be used.')
    parser.add_argument('--visual_extractor_pretrained', type=bool, default=True, help='whether to load the pretrained visual extractor')

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=2048, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=512, help='the dimension of the patch features.')# for mimic_cxr let "default=1536"
    parser.add_argument('--num_heads', type=int, default=8, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')


    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=0, help='whether decoding constraint.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=int, default=1, help='the number of gpus to be used.')
    parser.add_argument('--epochs', type=int, default=100, help='the number of training epochs.')
    parser.add_argument('--save_dir', type=str, default='/checkpoints', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=50, help='the patience of training.')
     # for Relational Memory
    parser.add_argument('--rm_num_slots', type=int, default=3, help='the number of memory slots.')
    parser.add_argument('--rm_num_heads', type=int, default=8, help='the numebr of heads in rm.')
    parser.add_argument('--rm_d_model', type=int, default=512, help='the dimension of rm.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ve', type=float, default=5e-5, help='the learning rate for the visual extractor.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument('--amsgrad', type=bool, default=True, help='.')

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')

    # Others
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')

    args= parser.parse_args()
    return args


def generate_report(frontal_image, lateral_image):
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    image1 = Image.open(io.BytesIO(frontal_image)).convert('RGB')
    image2 = Image.open(io.BytesIO(lateral_image)).convert('RGB')
    image1 = transform(image1)
    image2 = transform(image2)
    images = torch.stack([image1, image2], dim=0)  # Shape: [2, 3, 224, 224]
    images = images.unsqueeze(0)  # Shape: [1, 2, 3, 224, 224]
    
    print(f"Images shape: {images.shape}")  # Debugging print

    args = parse_agrs()
    tokenizer = Tokenizer(args)
    model = R2GenModel(args, tokenizer)

    try:
        data = torch.load(r'.\checkpoints\current_checkpoint.pth', map_location=torch.device('cpu'))
        model.load_state_dict(data['state_dict'])
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")  # Error handling

    try:
        output = model(images, mode='sample')
        reports = model.tokenizer.decode_batch(output.cpu().numpy())
    except Exception as e:
        print(f"Error generating report: {e}")  # Error handling
  
    print(f"Reports: {reports}")  # Debugging print
    return reports

@app.route('/')
def index():
    return render_template('index0.html')
@app.route('/upload', methods=['POST'])
def upload_files():
    if 'frontal' not in request.files or 'lateral' not in request.files:
        return jsonify({'error': 'Please upload both images'}), 400

    frontal_image = request.files['frontal'].read()
    lateral_image = request.files['lateral'].read()

    report = generate_report(frontal_image, lateral_image)

    return jsonify({'report': report})


    


###############################################################################
################################################################################

if __name__ == "__main__":
    port = 5000
    url = f"http://127.0.0.1:{port}"
    threading.Timer(1.25, lambda: webbrowser.open(url)).start()
    app.run(port=port, debug=False)

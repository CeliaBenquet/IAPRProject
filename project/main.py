import argparse
from utils import *
from Net import Net
import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim


data_base_path = os.path.join(os.pardir, 'data')
data_folder = 'lab-03-data'
data_part2_folder = os.path.join(data_base_path, data_folder, 'part2')


parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--input',
                    type = str, default = "./data/robot_parcours_1.avi",
                    help = 'Path to input file')

parser.add_argument('--output',
                    type = str, default = None,
                    help = 'Path to output file')

parser.add_argument('--training',
                    action = 'store_true', default = False,
                    help = 'Train the model for digits recognition on the MNIST dataset')

parser.add_argument('--display_training',
                    action = 'store_true', default = False,
                    help = 'If training, print performances')

parser.add_argument('--model_digits',
                    type = str, default = 'model/model_MNIST.pt',
                    help = 'Path to trained model for digits')

parser.add_argument('--epochs',
                    type = int, default = 25,
                    help = 'Number of epochs for the training (default:25)')

parser.add_argument('--mnist_data',
                    type = str, default = data_part2_folder,
                    help = 'Path to the MNIST data')

args = parser.parse_args()

## create images from the video &&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&&
create_data(args.input)

## generate and train model for digits recognition &&&&&&&&&&&&&&&&&&&&&&&&&&
if args.training: 
    #train the parameters of the model on the dataset
    train_model(args.model_digits, args.epochs, args.display_training, args.mnist_data)

if os.path.exists(args.model_digits):
    model=Net()
    model.load_state_dict(torch.load(args.model_digits))
    model.eval()
    print('hello you made it here')
    #use output=model(input) to use the model 


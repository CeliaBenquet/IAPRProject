import argparse
from utils import create_data

parser = argparse.ArgumentParser(description='Project 1 - Classification.')

parser.add_argument('--input',
                    type = str, default = "./data/robot_parcours_1.avi",
                    help = 'Path to input file')

parser.add_argument('--output',
                    type = str, default = None,
                    help = 'Path to output file')

args = parser.parse_args()

## create images from the video 
create_data(args)


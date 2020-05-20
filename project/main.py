import argparse
from utils import train_model, evaluate_expression
from video import *
import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim

import os


def process_video(input_path):
    # Read the video from specified path
    cam = cv2.VideoCapture(input_path)

    currentframe = 0
    cv2.namedWindow("bbs")

    all_centroids = []
    all_patches = []
    symbol_ids = []

    while(True): # reading from source
        ret, frame = cam.read()

        if ret:
            normalized = normalize(frame)

            thresholdedArrow = thresholdArrow(normalized).astype('uint8') * 255

            arrowBbox = bounding_box(thresholdedArrow)
            arrowBbox = expandBbox(arrowBbox, 20)

            robotCenter = get_center(arrowBbox)

            thresholded = threshold(normalized).astype('uint8') * 255
            bbs = [expandBbox(bb, 2) for bb in bounding_boxes(thresholded) if not isOverlapping(bb, arrowBbox)]

            centroids = [get_center(b) for b in bbs]

            patches = extractPatches(thresholded, bbs)
            patches = normalizePatches(patches)

            if currentframe == 0:
                all_centroids = centroids
                all_patches = patches
            elif len(patches) != len(all_centroids):
                distances = [np.linalg.norm(c - robotCenter) for c in all_centroids]
                min_distance_id = np.argmin(distances)
                if len(symbol_ids) == 0 or min_distance_id != symbol_ids[-1]:
                    symbol_ids.append(min_distance_id)


            cv2.rectangle(frame, arrowBbox[0:2], arrowBbox[2:4], (0, 0, 255))

            draw_bbs(frame, bbs)

            cv2.imshow("bbs", frame)

            cv2.waitKey()

            currentframe += 1
        else:
            break

    #The expression is here
    symbols = [all_patches[i] for i in symbol_ids]
    #printPatches(symbols)

    print(symbols)
    expression_value = evaluate_expression(symbols) 

    print(expression_value)

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()



if __name__ == '__main__':

    data_base_path = os.path.join(os.pardir, 'data')
    data_folder = 'lab-03-data'
    data_part2_folder = os.path.join(data_base_path, data_folder, 'part2')
    data_op_folder = os.path.join(data_base_path, 'data_operators')



    parser = argparse.ArgumentParser(description='Project 1 - Classification.')

    parser.add_argument('--input',
                        type = str, default = './data/robot_parcours_1.avi',
                        help = 'Path to input file')

    parser.add_argument('--output',
                        type = str, default = None,
                        help = 'Path to output file')

    parser.add_argument('--training',
                        action = 'store_true', default = False,
                        help = 'Train the model for digits recognition on the MNIST dataset')

    parser.add_argument('--run',
                        action = 'store_true', default = True,
                        help = 'Run from video')

    parser.add_argument('--display_training',
                        action = 'store_true', default = False,
                        help = 'If training, print performances')

    parser.add_argument('--model_digits',
                        type = str, default = 'model/model_MNIST.pt',
                        help = 'Path to trained model for digits')
                        
    parser.add_argument('--model_operators',
                        type = str, default = 'model/model_operators.pt',
                        help = 'Path to trained model for operators')

    parser.add_argument('--epochs',
                        type = int, default = 50,
                        help = 'Number of epochs for the training (default:25)')

    parser.add_argument('--mnist_data',
                        type = str, default = data_part2_folder,
                        help = 'Path to the MNIST data')

    args = parser.parse_args()


    ## generate and train model for digits recognition 
    if args.training: 
        #train the parameters of the model for digits 
        #train_model(args.model_digits, args.epochs, args.display_training, args.mnist_data, digits=True)
        #train the parameters of the model for operators 
        train_model(args.model_operators, args.epochs, args.display_training, data_op_folder, digits=False)

    if args.run:
        process_video(args.input)


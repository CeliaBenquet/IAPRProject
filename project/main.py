import argparse
from utils import *
from video import *
from Net import Net
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
    cv2.namedWindow("video")
    cv2.namedWindow("bbs")

    while(True):
        # reading from source
        ret, frame = cam.read()

        if ret:
            normalized = normalize(frame)

            thresholdedArrow = thresholdArrow(normalized).astype('uint8') * 255

            arrowBbox = bounding_box(thresholdedArrow)
            arrowBbox = expandBbox(arrowBbox, 20)

            cv2.rectangle(frame, arrowBbox[0:2], arrowBbox[2:4], (0, 0, 255))

            thresholded = threshold(normalized).astype('uint8') * 255
            bbs = [expandBbox(bb, 2) for bb in bounding_boxes(thresholded) if not isOverlapping(bb, arrowBbox)]

            patches = extractPatches(thresholded, bbs)
            patches = normalizePatches(patches)

          #  printPatches(patches)


            # TODO: Classify patches



            cv2.imshow("video", thresholded)
            draw_bbs(frame, bbs)

            cv2.imshow("bbs", frame)

            cv2.waitKey()

            # increasing counter so that it will
            # show how many frames are created
            currentframe += 1
        else:
            break

    # Release all space and windows once done
    cam.release()
    cv2.destroyAllWindows()




if __name__ == '__main__':

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

    parser.add_argument('--eval',
                        action = 'store_true', default = False,
                        help = 'Evaluate the model')

    parser.add_argument('--run',
                        action = 'store_true', default = True,
                        help = 'Run from video')

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

    ## generate and train model for digits recognition &&&&&&&&&&&&&&&&&&&&&&&&&&
    if args.training: 
        #train the parameters of the model on the dataset
        train_model(args.model_digits, args.epochs, args.display_training, args.mnist_data)

    if args.eval:
        if os.path.exists(args.model_digits):
            model=Net()
            model.load_state_dict(torch.load(args.model_digits))
            model.eval()
            print('hello you made it here')
            #use output=model(input) to use the model 

    if args.run:
        process_video(args.input)


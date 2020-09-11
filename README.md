# IAPR Project
Image Analysis and Pattern Recognition

The course aims at learning tools to analyse images and recognize objects in it.


The main task of the project is to find the result of an equation based on a video sequence. The equation is indicated by a moving robot. The detailed scenario is defined as follows:
- Several mathematical operators (multiplication, division, minus, plus, equal) are placed on the table. The color of operators is blue .
- Several handwritten digits (0 to 8) are placed on the table. Digits are always black. 
- From an initial location somewhere on the table, the robot moves around the table. Each time the robot passes above an operator or a digit, the symbol located below the robot is added to the equation. For example the sequence “2” → “+” → “3” → “=” becomes “2+3=”.
- The sequence always starts with a digit and ends with the operator “=”.

## Usage
Run the following command from the src directory:
```bash
python main.py

Options:
    --run                          Run from input video.
                                      (default mode)
    --input STR                    Path to the input video.
                                      (defaut: train video "..\data\robot_parcours_1.avi")
    --output STR                   Path to the output video 
                                      similar to input video with the analysis and expression in more.
                                      (default: "output.avi", in same folder)
    --training_digits              Train model for digits recognition on MNIST dataset.
    --training_operators           Train model for operators recognition.
    --display_training             Display performances on train and test set. 
                                        (if used with --rotation, trains the rotation invariant model)
    --model_digits STR             Path to trained model for digits recognition. 
                                       (default: "model/model_MNIST.pt")
    --model_digits_rotation STR    Path to trained model for digits recognition invariant to rotation 
                                       (default: "model/model_MNIST_rotation.pt")
    --model_operators STR          Path to trained model for operators recognition. 
                                       (default: "model/model_operators.pt")
    --epochs INT                   Number of epochs performed at training.
                                       (default: 50)
    --mnist_data STR               Path to the MNIST dataset folder.
                                       (default: ".\data\lab-03-data\part2")
    --operators_data STR           Path to operators images, already treated. 
                                       (correspond to original images + images cropped from video)
    --rotation                     Use the digits model invariant to rotation. 
                                       if used in combination with --training_digits, trains the rotation invariant model.
```

By default, the script works on the video to: extract the symbols in order, recognize them and evaluate the result.

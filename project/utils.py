import gzip
import numpy as np
import matplotlib.pyplot as plt
from Net import Net
import torch 
from torch import Tensor 
from torch import nn
from torch.nn import functional as F
from torch import optim
from scipy import ndimage
import skimage.morphology
import cv2
import os
import imgaug.augmenters as iaa
from sklearn.model_selection import train_test_split
import random
from skimage.filters import threshold_otsu
import operator

def extract_data(filename, image_shape, image_number):
    """
    Extract train/test data 
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data

def extract_labels(filename, image_number):
    """
    Extract train/test labels
    """
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def rotate(data, degrees):
    """
    Rotate images in data by a given degree
    """
    data = data.copy()
    for image in data[:]:
        image = ndimage.rotate(image, degrees, reshape=False)
    return data

def augment_dataset(data, labels, rotation):
    """
    Augment MNIST data set by rotating the images by step of 10° and transfo to array
    model resistant to random orientation of the images 
    """

    rotated_data = [data]
    rotated_labels = [labels]

    #rotation only if rotation is set to True 
    if rotation:
        for degrees in range(10, 360, 10):
            rotated_data.append(rotate(data, degrees))
            rotated_labels.append(labels)

    #transfo from list to array
    data = np.concatenate(rotated_data)
    labels = np.concatenate(rotated_labels)

    return data, labels

def create_mnist_data(data_dir, mini_batch_size, rotation):
    """ 
    Create the dataset to train on digits 
    data_dir: directoy where MNIST data set is stored
    mini_batch_size: batch size for the Net used to set the train/test set size 
    rotation: if set to True, dataset to train for rotation invariance
    """
    image_shape = (28, 28)
    if rotation: 
        print("Augmentation of MNIST dataset to be invariant to rotation...")
        #initial dataset size, will increase with augmentation
        train_set_size = 1500
        test_set_size = 400
    else: 
        train_set_size = 20000
        test_set_size = 6000

    #path to dataset 
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    #extract data and labels
    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)

    #augment data by adding rotation robustness , only if rotation set to True
    train_images, train_labels = augment_dataset(train_images,train_labels, rotation)
    test_images, test_labels = augment_dataset(test_images,test_labels, rotation)

    #remove class 9, resize features, convert to Tensor 
    train_images, train_labels = preprocessing_mnist(train_images,train_labels, mini_batch_size)
    test_images, test_labels = preprocessing_mnist(test_images,test_labels, mini_batch_size)

    #display dataset sizes 
    print("Training set size -----> ", train_images.size())
    print("Training labels size --> ", train_labels.size())
    print("Testing set size ------> ", test_images.size())
    print("Testing labels size ---> ",test_labels.size())

    #add dimensionalities to fit requirements for net input 
    return train_images[:,None,:,:], test_images[:,None,:,:], train_labels, test_labels

def preprocessing_mnist(data, labels, mini_batch_size): 
    """
    Preprocess datasets to be optimized for training/testing
    """
    #remove class of 9
    data = data[labels != 9]
    labels = labels[labels != 9]

    #resize to be divisable by mini_batch_size in Net
    data_size=data.shape[0]
    mod=data_size%mini_batch_size 

    if mod != 0: 
        data = data[: data_size-mod]
        labels = labels[: data_size-mod]

    #convert to Tensors 
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    return data, labels

def chooseImage(data, labels, label):
    #label: the class
    #choose randomly between image from video original image 
    images_idx = [i for i, e in enumerate(labels) if e == label]
    return data[random.choice(images_idx)]

def augment_data_imgaug(data, labels, nb_op):
    """
    Augment operators dataset to have more images + be resistant to different kind of transfo 
    """
    data_aug, labels_aug = [], []

    #define the possible transformations 
    seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontal flips
    iaa.Crop(percent=(0, 0.1)), # random crops
    iaa.LinearContrast((0.75, 1.5)), # strengthen or weaken the contrast in each image
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05*255), per_channel=0.5), # Add gaussian noise.
    iaa.Multiply((0.8, 1.2), per_channel=0.2), # Make some images brighter and some darker.
    iaa.Affine(   # Apply affine transformations to each image.
        scale={"x": (0.8, 1.2), "y": (0.8, 1.2)},
        translate_percent={"x": (-0.2, 0.2), "y": (-0.2, 0.2)},
        rotate=(-90, 90),
        shear=(-8, 8) 
    )], random_order=True) # apply augmenters in random order

    #nb of sample per class (per operator)
    sample_per_class=1000

    for batch_idx in range(int(sample_per_class)): 
        for i in range(nb_op): 
            data_aug.append(seq.augment_images(chooseImage(data, labels, i))) #add one image
            labels_aug.append(i) #add the corresponding label

    return data_aug, labels_aug


def create_operators_data(data_dir):
    """
    Create train/test sets for the operators
    data_dir: directory where the original operators images are stored
    the operators are stored as numbers to be classified
    0: =
    1: *
    2: /
    3: +
    4: -
    """
    #nb of operators 
    nb_op=5
    
    #get the data
    data,labels = [],[]
    for i in range(nb_op) :

        #image from the training video (don't have it for minus)
        img_path_main = os.path.join(data_dir, str(i), 'main.jpg')
        if os.path.isfile(img_path_main):
            temp_img_main = cv2.imread(img_path_main)
            temp_img_main = thresholding(cv2.cvtColor(temp_img_main, cv2.COLOR_BGR2GRAY))
            labels.append(i)
            data.append(temp_img_main)

        #image from original operators 
        img_path_or = os.path.join(data_dir, str(i), 'original.jpg')
        if os.path.isfile(img_path_or):
            temp_img = cv2.imread(img_path_or)
            temp_img = thresholding(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY))
            labels.append(i)
            data.append(temp_img) 

    #augment the data
    print("Augmentation of the data for the operators...")
    data_aug, labels_aug = augment_data_imgaug(data, labels, nb_op)
    
    #split dataset in train and test 
    train_images, test_images, train_labels, test_labels = train_test_split(data_aug, labels_aug, \
                                                    test_size=0.3, random_state=42)

    #transform to Tensor (PyTorch) and flattening
    train_images, train_labels = preprocessing_op(train_images,train_labels)
    test_images, test_labels = preprocessing_op(test_images,test_labels)

    #diplay size of the datasets
    print("Training set size -----> ", train_images.size())
    print("Training labels size --> ", train_labels.size())
    print("Testing set size ------> ", test_images.size())
    print("Testing labels size ---> ",test_labels.size())

    #add dimensions to fir the net requirements 
    return train_images[:,None,:,:], test_images[:,None,:,:], train_labels, test_labels


def preprocessing_op(data, labels): 
    """
    Preprocess the operators images to fit the requirements for the net 
    resize images to 28*28 and transformation to tensors 
    """
    data_resized=[]
    for image in data: 
        #resize 
        data_resized.append(cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA))
    #turn into tensors 
    return (torch.FloatTensor(data_resized), torch.LongTensor(labels)) 


def train_model(path_model, epochs, display_perf, data_dir, digits=True, rotation=False):
    """
    Main function for training 
    Used for both digits (with or without rotation) or operators training 
        path_model: file to which the output model is stored 
        epochs: number of epochs for the training 
        display_perf: if True, display graphs of loss and accuracy over the epochs for train and test 
        data_dir: directory where the data are stored 
        digits: if True, model for digits recognition is trained else model for operators recognition
        rotation: if True, model for digits recognition invariant to rotation is trained, else no rotation invariance  
    """ 
    #conditions for the net 
    torch.manual_seed(0)
    np.random.seed(0)
    mini_batch_size = 50 

    #generate data 
    if digits:
        #train model for digits recognition 
        #if rotation is True: digit recognition invariant to rotation 
        train_input, test_input, train_target, test_target = create_mnist_data(data_dir, mini_batch_size, rotation)
        #number of classes (9 digits - no 9)
        n_output = 9
        print("Start training model on MNIST dataset...")

    else: 
        #train model for operators recognition 
        train_input, test_input, train_target, test_target = create_operators_data(data_dir)
        #number of classes (5 operators)
        n_output = 5
        print("Start training model on operators dataset...")

    #create net and parameters of the model 
    model = Net(n_output)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001) 
    display_step = 5

    #stock performances to display 
    losses_tr = []
    accuracies_tr = []
    losses_te = []
    accuracies_te = []
    
    #best accuracy on all epochs 
    best_acc=-100
    
    #needed as we use batch normalization and dropout regularization techniques 
    model.train()
    
    for e in range(epochs):

        #using mini-batches 
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()

            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))            
            loss.backward()
            
            optimizer.step()
          
        #results for training
        avg_loss_tr, avg_accuracy_tr = compute_performances(model, train_input, train_target, mini_batch_size, criterion)
        if e % display_step ==0:            
            print('Epoch: %02d' %(e), '--> train loss = ' + "{:.3f}".format(avg_loss_tr), ', train accuracy: ' + "{:.3f}".format(avg_accuracy_tr) + "%")
        losses_tr.append(avg_loss_tr)
        accuracies_tr.append(avg_accuracy_tr)
        
        #results for testing
        avg_loss_te, avg_accuracy_te = compute_performances(model, test_input, test_target, mini_batch_size, criterion)
        if e % display_step ==0:            
            print('          --> test loss = ' + "{:.3f}".format(avg_loss_te), ', test accuracy: ' + "{:.3f}".format(avg_accuracy_te) + "%")
            print('-------------------------------------------------------------')
        losses_te.append(avg_loss_te)
        accuracies_te.append(avg_accuracy_te)

        #keep best model as output model
        if avg_accuracy_te > best_acc: 
            best_model=model.state_dict()
            best_acc=avg_accuracy_te

    #if True, display graphs and print the best model (the one that is saved)
    if display_perf:   
        #print best model 
        print('Best accuracy on test: {}'.format(best_acc))

        # plot the accuracy and loss
        x_axis=range(0,epochs,5)
        plt.figure(figsize = (20,10))
        plt.subplot(221)
        plt.plot(range(epochs), losses_tr, color='r')
        plt.xticks(x_axis)
        plt.title("Average Loss at training")
        plt.subplot(222)
        plt.plot(range(epochs), accuracies_tr)
        plt.xticks(x_axis)
        plt.title("Accuracy at training (in %)")
        plt.subplot(223)
        plt.plot(range(epochs), losses_te, color='r')
        plt.xticks(x_axis)
        plt.title("Average Loss at testing")
        plt.subplot(224)
        plt.plot(range(epochs), accuracies_te)
        plt.xticks(x_axis)
        plt.title("Accuracy at testing (in %)")
        plt.show()

    # save best model on all epochs  
    try: 
    
        # creating a folder named data 
        if not os.path.exists('model'): 
            os.makedirs('model') 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory for trained model') 

    # save model for epoch that gives best accuracy on test
    torch.save(best_model,path_model)    
    print("Training done... the model was saved.")


def compute_performances(model, inputs, labels, mini_batch_size, criterion):
    """
    Compute loss and accuracy on train or test model depending on the input
    the function is called at each epoch of the training to get progressions 
    model: the model at the time of the call
    """
    model.eval()
    
    sum_loss=0.0
    correct=0.0
    
    for b in range(0, inputs.size(0), mini_batch_size):
        # get the outputs from the trained model
        output = model(inputs.narrow(0, b, mini_batch_size))

        loss = criterion(output, labels.narrow(0, b, mini_batch_size))
        sum_loss += loss.item()
        
        #prediction based on output: takes the higher proba 
        predicted = torch.argmax(output, 1)
        #check if the predictions corresponds to the label
        correct += (predicted == labels.narrow(0, b, mini_batch_size)).sum().double()
    
    #calculate the averaged loss and accuracy over the epoch (on all mini-batches)
    avg_loss=sum_loss/inputs.shape[0]
    avg_acc=100 * correct/inputs.shape[0]
    
    return avg_loss, avg_acc
    

def evaluate_expression(symbols, args, rotation=False):
    """
    Evaluate given images and return label as a charactere 
        symbols: can be one or a set of images 
        rotation: if True, use the roation invariant model for digits 
    """
    # we know that first is a digits and second is an operator and so forth 
    expression_value=""

    #go through the list of images to evaluate 
    for symb in symbols: 

        #if first character or last one was operator => current is a digit
        if (not expression_value) or (not expression_value[-1].isdigit()):
            n_output=9
            if rotation: 
                # classification as a digit, invariant to rotation
                if os.path.exists(args.model_digits_rotation):
                    model=Net(n_output)
                    #load the corresponding model 
                    model.load_state_dict(torch.load(args.model_digits_rotation))
                    model.eval()
                    #evaluate the label of the image 
                    output = model(preprocessing_symb(symb))
                    predicted = torch.argmax(output)
                    #transfo to str (we know it's a digit)
                    char=str(int(predicted))
            else: 
                # classification as a digit, no invariance to rotation 
                if os.path.exists(args.model_digits):
                    model=Net(n_output)
                    #load the corresponding model 
                    model.load_state_dict(torch.load(args.model_digits))
                    model.eval()
                    #evaluate the label of the image
                    output = model(preprocessing_symb(symb))
                    predicted = torch.argmax(output)
                    #transfo to str (we know it's a digit )
                    char=str(int(predicted))
            
        # last character was a digits => current is an operator 
        else: 
            # classification as an operator 
            n_output=5
            if os.path.exists(args.model_operators):
                model=Net(n_output)
                #load the corresponding model
                model.load_state_dict(torch.load(args.model_operators))
                model.eval()
                #evaluate the label of the image
                output = model(preprocessing_symb(symb))
                predicted = torch.argmax(output)
                #transfo to str knowing correspondance from label to operator 
                char=opToStr(int(predicted))

        # add the charactere to the equation
        expression_value+=char

    return expression_value


def calculate_equation(expression):
    """
    Calculate result of the equation 
    doesn't take last element into account (=)
    """
    return eval(expression[:-1])


def preprocessing_symb(symbol):
    """
    Transform symb to a Tensor to be evaluated by model 
    """
    symbol = torch.Tensor(symbol)
    return symbol[None,None,:,:]

def thresholding(image):
    """
    Otsu adaptative thresholding 
    """
    _,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return th2


def opToStr(op_int): 
    """
    Convert class label to corresponding operator in the exp
    0: =
    1: *
    2: /
    3: +
    4: -
    """
    if op_int == 0: 
        op_str = '='
    elif op_int == 1: 
        op_str = '*'
    elif op_int == 2: 
        op_str = '/'
    elif op_int == 3: 
        op_str = '+'
    elif op_int == 4: 
        op_str = '-'
    return op_str

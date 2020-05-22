import gzip
import numpy as np
import matplotlib.pyplot as plt
from Net import Net
from CNNet import CNNet, CNNet2
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
    with gzip.open(filename) as bytestream:
        bytestream.read(16)
        buf = bytestream.read(np.prod(image_shape) * image_number)
        data = np.frombuffer(buf, dtype=np.uint8).astype(np.float32)
        data = data.reshape(image_number, image_shape[0], image_shape[1])
    return data

def extract_labels(filename, image_number):
    with gzip.open(filename) as bytestream:
        bytestream.read(8)
        buf = bytestream.read(1 * image_number)
        labels = np.frombuffer(buf, dtype=np.uint8).astype(np.int64)
    return labels

def rotate(data, degrees):
    data = data.copy()
    for image in data[:]:
        image = ndimage.rotate(image, degrees, reshape=False)
    return data

def augment_dataset(data, labels):
    rotated_data = [data]
    rotated_labels = [labels]
    for degrees in range(10, 360, 10):
        rotated_data.append(rotate(data, degrees))
        rotated_labels.append(labels)
        #print("Generated data with", degrees, "degrees")

    #print("Rotated data size ", len(rotated_data))
    #print("Rotated labels size ", len(rotated_labels))
    data = np.concatenate(rotated_data)
    labels = np.concatenate(rotated_labels)
    #print("data shape ", data.shape)
    return data, labels

def create_mnist_data(data_dir, mini_batch_size):
    image_shape = (28, 28)
    #initial dataset size, will increase with augmentation
    train_set_size = 500
    test_set_size = 150

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

    #augment data by adding rotation robustness 
    print("Augmentation of MNIST dataset to be resistant to rotation...")
    train_images, train_labels = augment_dataset(train_images,train_labels)
    test_images, test_labels = augment_dataset(test_images,test_labels)

    #remove class 9, resize features, convert to Tensor 
    train_images, train_labels = preprocessing_mnist(train_images,train_labels, mini_batch_size)
    test_images, test_labels = preprocessing_mnist(test_images,test_labels, mini_batch_size)

    #display dataset sizes 
    print(train_images.size())
    print(train_labels.size())
    print(test_images.size())
    print(test_labels.size())

    return train_images[:,None,:,:], test_images[:,None,:,:], train_labels, test_labels

def preprocessing_mnist(data, labels, mini_batch_size): 
    #remove class of 9
    data = data[labels != 9]
    labels = labels[labels != 9]
    
    #resize to be divisable by mini_batch_size 
    data_size=data.shape[0]
    mod=data_size%mini_batch_size 
    if mod != 0: 
        data = data[: data.shape[0]-mod]
        labels = labels[: labels.shape[0]-mod]
    
    #convert to Tensors 
    data = torch.from_numpy(data)
    labels = torch.from_numpy(labels)

    return data, labels

def chooseImage(data, labels, label):
    #label: the class
    #choose randomly between image from video original image 
    #img = data[random.choice([labels==label])]
    images_idx = [i for i, e in enumerate(labels) if e == label]
    return data[random.choice(images_idx)]

def augment_data_imgaug(data, labels, nb_op):
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

    #apply possible transformation to each class 
    sample_per_class=700  #nb images for training/testing

    for batch_idx in range(int(sample_per_class)): #nb of sample per class 
        for i in range(nb_op): 
            data_aug.append(seq.augment_images(chooseImage(data, labels, i))) #add one image
            labels_aug.append(i) #add the corresponding label

    return data_aug, labels_aug


def create_operators_data(data_dir):
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

            #cv2.imshow('img', temp_img_main)
            #cv2.waitKey(0)
            #cv2.destroyAllWindows()

        #image from original operators 
        img_path_or = os.path.join(data_dir, str(i), 'original.jpg')
        if os.path.isfile(img_path_or):
            temp_img = cv2.imread(img_path_or)
            temp_img = thresholding(cv2.cvtColor(temp_img, cv2.COLOR_BGR2GRAY))
            labels.append(i)
            data.append(temp_img) 
        
        #cv2.imshow('img', temp_img)
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()

    #augment the data
    print("Augmentation of the data for the operators...")
    data_aug, labels_aug = augment_data_imgaug(data, labels, nb_op)
    
    #split dataset in train and test 
    train_images, test_images, train_labels, test_labels = train_test_split(data_aug, labels_aug, \
                                                    test_size=0.3, random_state=42)

    #transform to Tensor (PyTorch) and flattening
    train_images, train_labels = preprocessing_op(train_images,train_labels)
    test_images, test_labels = preprocessing_op(test_images,test_labels)

    print(train_images.size())
    print(train_labels.size())
    print(test_images.size())
    print(test_labels.size())

    return train_images[:,None,:,:], test_images[:,None,:,:], train_labels, test_labels


def preprocessing_op(data, labels): 
    data_resized=[]
    for image in data: 
        #cv2.imshow('img', cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA))
        #cv2.waitKey(0)
        #cv2.destroyAllWindows()
        data_resized.append(cv2.resize(image, (28,28), interpolation = cv2.INTER_AREA))
    return (torch.FloatTensor(data_resized), torch.LongTensor(labels)) 


def train_model(path_model, epochs, display_perf, data_dir, digits=True):

    #conditions for the net 
    torch.manual_seed(0)
    np.random.seed(0)
    mini_batch_size = 50 

    #generate data 
    if digits:
        train_input, test_input, train_target, test_target = create_mnist_data(data_dir, mini_batch_size)
        n_output = 9
        print("Start training model on MNIST dataset...")

    else: 
        train_input, test_input, train_target, test_target = create_operators_data(data_dir)
        n_output = 5
        print("Start training model on operators dataset...")

    #create net and parameters of the model 
    model = CNNet(n_output)
    #model = Net(n_input, n_hidden, n_output)
    criterion = nn.CrossEntropyLoss()
    display_step = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    #stock performances to display 
    losses_tr = []
    accuracies_tr = []
    losses_te = []
    accuracies_te = []
    
    best_acc=0
    
    model.train()
    
    for e in range(epochs):

        # We do this with mini-batches
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
        losses_te.append(avg_loss_te)
        accuracies_te.append(avg_accuracy_te)

        #keep best model 
        if avg_accuracy_te > best_acc: 
            best_model=model.state_dict()
            best_acc=avg_accuracy_te

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

    # save model
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
    
    model.eval()
    
    sum_loss=0.0
    correct=0.0
    
    for b in range(0, inputs.size(0), mini_batch_size):
        # get the outputs from the trained model
        output = model(inputs.narrow(0, b, mini_batch_size)) #should have done it by mini batches 

        loss = criterion(output, labels.narrow(0, b, mini_batch_size))
        sum_loss += loss.item()
        
        predicted = torch.argmax(output, 1)
        correct += (predicted == labels.narrow(0, b, mini_batch_size)).sum().double()
    
    avg_loss=sum_loss/inputs.shape[0]
    avg_acc=100 * correct/inputs.shape[0]
    
    return avg_loss, avg_acc
    

def evaluate_expression(symbols, args):
    # we know that first is a digits and second is an operator and so forth 
    expression_value=""

    for symb in symbols: 

        if (not expression_value) or (not expression_value[-1].isdigit()) : #if first character or last one was operator
            # classification as a digit 
            if os.path.exists(args.model_digits):
                model=CNNet(9)
                model.load_state_dict(torch.load(args.model_digits))
                model.eval()
                output = model(preprocessing_symb(symb))
                #print(output)
                predicted = torch.argmax(output)
                char=str(int(predicted))
        
        else: # last character was a digits
            # classification as an operator 
            if os.path.exists(args.model_operators):
                model=CNNet(5)
                model.load_state_dict(torch.load(args.model_operators))
                model.eval()
                preprocessing_symb(symb)
                output = model(preprocessing_symb(symb))
                predicted = torch.argmax(output)
                char=opToStr(int(predicted))

        # add the charactere to the equation
        expression_value+=char

    print('Equation: ', expression_value)

    result = calculate_equation(expression_value)

    return result 


def calculate_equation(expression):
    result = expression[0]

    for i in range(1,len(expression)-1,2): 
        oper=expression[i]
        digit=expression[i+1]
        if oper=='=' and i!=expression[-1]:
            raise NameError('Operator \'=\' was detected before the end of the expression')
        result = eval_binary_expr(result,oper,digit)

    return result 

def eval_binary_expr(op1, oper, op2,
                     get_operator_fn={
                         '+' : operator.add,
                         '-' : operator.sub,
                         '*' : operator.mul,
                         '/' : operator.truediv,
                         }.get):
    # calculate result for binary expression 
    op1,op2 = int(op1), int(op2)
    return get_operator_fn(oper)(op1, op2)


def preprocessing_symb(symbol):
    symbol = torch.FloatTensor(thresholding(symbol))
    return symbol[None,None,:,:]

def thresholding(image):
    #using otsu adaptative thresholding 
    _,th2 = cv2.threshold(image,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    return th2


def opToStr(op_int): 
    #convert classes to corresponding operator in the exp
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


def preprocessing_test(symbol):
    symbol = torch.FloatTensor(symbol)
    return symbol[None,None,:,:]

def test_model(args, digits=True):
    if digits:
        train_input, test_input, train_target, test_target = create_mnist_data(args.mnist_data, 50)
        n_output = 9
        model=CNNet(9)
        model.load_state_dict(torch.load(args.model_digits))
        model.eval()

    else: 
        train_input, test_input, train_target, test_target = create_operators_data(args.operators_data)
        n_output = 5
        model=CNNet(5)
        model.load_state_dict(torch.load(args.model_operators))
        model.eval()

   # obtain one batch of test images
    nb_im=20
    prng = np.random.RandomState(seed=123456789)  # seed to always re-draw the same distribution
    plt_ind = prng.randint(low=0, high=test_input.shape[0], size=nb_im)

    images=[]
    labels=[]
    for i in plt_ind:
        images.append(test_input[i])
        labels.append(test_target[i])

    #images, labels = np.array(images),np.array(labels)
    images, labels = torch.stack(images), torch.stack(labels)
    print(images.size())

    #images_,labels_=preprocessing(images, labels)

    # get sample outputs
    output = model(images.float())
    # convert output probabilities to predicted class
    preds = torch.argmax(output, 1)


    # plot the images in the batch, along with predicted and true labels
    images=images.numpy()
    fig = plt.figure(figsize=(25, 4))
    for idx in range(nb_im):
        ax = fig.add_subplot(nb_im/10, 10, idx+1, xticks=[], yticks=[])
        ax.imshow(images[idx], cmap='gray')
        ax.set_title("{} ({})".format(str(preds[idx].item()), str(labels[idx].item())),
                    color=("green" if preds[idx]==labels[idx] else "red"))
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
        print("Generated data with", degrees, "degrees")

    print("Rotated data size ", len(rotated_data))
    print("Rotated labels size ", len(rotated_labels))
    data = np.concatenate(rotated_data)
    labels = np.concatenate(rotated_labels)
    print("data shape ", data.shape)

    return data, labels



def create_mnist_data(n_input, data_dir):
    image_shape = (28, 28)
    train_set_size = 60000
    test_set_size = 10000

    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte.gz')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte.gz')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte.gz')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte.gz')

    #extract data and labels
    train_images = extract_data(train_images_path, image_shape, train_set_size)
    test_images = extract_data(test_images_path, image_shape, test_set_size)
    train_labels = extract_labels(train_labels_path, train_set_size)
    test_labels = extract_labels(test_labels_path, test_set_size)

    train_images, train_labels = augment_dataset(train_images, train_labels)
    test_images, test_labels = augment_dataset(test_images, test_labels)

    print("------------")
    print(train_labels.shape)   
    print(test_labels.shape)   

    #transform to Tensor (PyTorch) and flattening
    train_images, train_labels = preprocessing(train_images,train_labels,n_input)
    test_images, test_labels = preprocessing(test_images,test_labels,n_input)


    print(train_images.size())
    print(train_labels.size())
    print(test_images.size())
    print(test_labels.size())

    return train_images, test_images, train_labels, test_labels



def preprocessing(data, labels, n_input): 
    data = data[labels != 9]
    labels = labels[labels != 9]
    return (torch.from_numpy(data).view(-1, n_input), torch.from_numpy(labels))

def train_model(path_model, epochs, display_perf, data_dir):
    
    #conditions for the net 
    n_input = 784
    n_hidden = 100
    n_output = 10
    torch.manual_seed(0)
    np.random.seed(0)

    #generate data 
    train_input, test_input, train_target, test_target = create_mnist_data(n_input, data_dir)

    #create net and parameters of the model 
    model = Net(n_input, n_hidden, n_output)
    criterion = nn.CrossEntropyLoss()
    mini_batch_size = 838
    display_step = 5
    optimizer = optim.Adam(model.parameters(), lr=0.001) 

    #stock performances to display 
    losses_tr = []
    accuracies_tr = []
    losses_te = []
    accuracies_te = []
    
    print("Start training model on MNIST dataset...")
    
    model.train()
    
    for e in range(epochs):

        # We do this with mini-batches
        for b in range(0, train_input.size(0), mini_batch_size):
            optimizer.zero_grad()
            
            output = model(train_input.narrow(0, b, mini_batch_size))

            loss = criterion(output, train_target.narrow(0, b, mini_batch_size))            
            loss.backward()
            
            optimizer.step() #equivalent of the for loop update 
          
        #results for training
        avg_loss_tr, avg_accuracy_tr = compute_performances(model, train_input, train_target, 838, criterion)
        if e % display_step ==0:            
            print('Epoch: %02d' %(e), '--> train loss = ' + "{:.3f}".format(avg_loss_tr), ', train accuracy: ' + "{:.3f}".format(avg_accuracy_tr) + "%")
        losses_tr.append(avg_loss_tr)
        accuracies_tr.append(avg_accuracy_tr)
        
        #results for testing
        avg_loss_te, avg_accuracy_te = compute_performances(model, test_input, test_target, 999, criterion)
        if e % display_step ==0:            
            print('          --> test loss = ' + "{:.3f}".format(avg_loss_te), ', test accuracy: ' + "{:.3f}".format(avg_accuracy_te) + "%")
        losses_te.append(avg_loss_te)
        accuracies_te.append(avg_accuracy_te)

    if display_perf:   
        # plot the accuracy and loss
        plt.figure(figsize = (20,10))
        plt.subplot(221)
        plt.plot(range(epochs), losses_tr, color='r')
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.title("Average Loss at training")
        plt.subplot(222)
        plt.plot(range(epochs), accuracies_tr)
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.title("Accuracy at training (in %)")
        plt.subplot(223)
        plt.plot(range(epochs), losses_te, color='r')
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.title("Average Loss at testing")
        plt.subplot(224)
        plt.plot(range(epochs), accuracies_te)
        plt.xticks([0, 5, 10, 15, 20, 25])
        plt.title("Accuracy at testing (in %)")
        plt.show()

    #save model
    try: 
    
        # creating a folder named data 
        if not os.path.exists('model'): 
            os.makedirs('model') 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory for trained model') 


    torch.save(model.state_dict(),path_model)    
    print("Training done... the model was saved.")


def compute_performances(model, inputs, labels, mini_batch_size, criterion):
    
    model.eval()
    
    sum_loss=0.0
    correct=0.0
    
    for b in range(0, inputs.size(0), mini_batch_size):
        #get the outputs from the trained model
        output = model(inputs.narrow(0, b, mini_batch_size)) #should have done it by mini batches 

        loss = criterion(output, labels.narrow(0, b, mini_batch_size))
        sum_loss += loss.item()
        
        predicted = torch.argmax(output, 1)
        correct += (predicted == labels.narrow(0, b, mini_batch_size)).sum().double()
    
    avg_loss=sum_loss/inputs.shape[0]
    avg_acc=100 * correct/inputs.shape[0]
    
    return avg_loss, avg_acc
    

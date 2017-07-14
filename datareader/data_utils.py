#coding:utf-8
import numpy as np
import struct
def load_MNIST_images(filename):
    binfile = open(filename , 'rb')
    buf = binfile.read()
    index = 0
    magic, numImages , numRows , numColumns = struct.unpack_from('>IIII' , buf , index)
    index += struct.calcsize('>IIII')
    images = []
    for i in range(numImages):
        image = struct.unpack_from('>784B',buf,index)
        index += struct.calcsize('>784B')
        images.append(list(image))
    images = np.array(images)
    N,_ = images.shape
    images = images.reshape(N,1,28,28).transpose(0,2,3,1).astype("float")
    return images    
def load_MNIST_labels(filename):
    binfile = open(filename , 'rb')
    buf = binfile.read()
    index = 0
    magic, numImages =struct.unpack_from('>II' , buf , index)
    index += struct.calcsize('>II')
    labels = []
    for i in range(numImages):
        label = struct.unpack_from('>1B',buf,index)
        index += struct.calcsize('>1B')
        labels.extend(list(label))
    return np.array(labels)
def get_MNIST_data(num_validation=1000):
    X_train_path = 'data/train-images-idx3-ubyte'    
    y_train_path = 'data/train-labels-idx1-ubyte'
    X_test_path = 'data/t10k-images-idx3-ubyte'
    y_test_path = 'data/t10k-labels-idx1-ubyte'
    X_train = load_MNIST_images(X_train_path)
    y_train = load_MNIST_labels(y_train_path)
    X_test = load_MNIST_images(X_test_path)
    y_test = load_MNIST_labels(y_test_path)
    mask = range(num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    X_train = X_train[len(mask):]
    y_train = y_train[len(mask):]
    #减均值
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image
    #转变维度
    X_train = X_train.transpose(0, 3, 1, 2).copy()
    X_val = X_val.transpose(0, 3, 1, 2).copy()
    X_test = X_test.transpose(0, 3, 1, 2).copy()  
    return {
        'X_train': X_train, 'y_train': y_train,
        'X_val': X_val, 'y_val': y_val,
        'X_test': X_test, 'y_test': y_test,
    }    
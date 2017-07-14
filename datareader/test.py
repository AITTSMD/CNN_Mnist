from data_utils import *
dd = get_MNIST_data()
print dd['X_train'].shape
print dd['X_val'].shape
print dd['X_test'].shape
print dd['y_train'].shape
print dd['y_val'].shape
print dd['y_test'].shape
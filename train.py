#coding:utf-8
from classifiers.cnn import *
from datareader.data_utils import *
from layers.layers import *
from layers.layer_utils import *
from solver.solver import *
from time import *
if __name__ == "__main__":
    #样本数据
    data = get_MNIST_data()
    #搭建模型
    num_cp =4
    num_hid =2
    num_pool = 2 
    num_filters = [32,32,64,64]
    num_hids = [128,128]
    filter_size = [3,3,3,3]
    conv_param = [{} for i in xrange(num_cp)]
    pool_param = [{} for i in xrange(num_pool)]
    for i in range(num_cp):
        conv_param[i]['stride'] = 1
        conv_param[i]['pad'] = (filter_size[i] - 1)/2
    for i in range(num_pool):
        pool_param[i]['pool_height'] = 2
        pool_param[i]['pool_width'] = 2
        pool_param[i]['stride'] = 2   
        
    MyModel = MultiLaterConvNet(input_dim=(1, 28, 28),
                                num_filters=num_filters, 
                                num_hids=num_hids, 
                                filter_size=filter_size, 
                                num_classes=10, 
                                weight_scale=1e-1, 
                                reg=0.01, 
                                conv_params=conv_param, 
                                pool_params=pool_param, 
                                dtype=np.float32, 
                                use_batchnorm=True)
    #num_train = 100
    input_data = {
        'X_train': data['X_train'],
        'y_train': data['y_train'],
        'X_val': data['X_val'],
        'y_val': data['y_val'],
    }
    print input_data['X_train'].shape
    print input_data['y_train'].shape
    print input_data['X_val'].shape
    print input_data['y_val'].shape
    solver = Solver(MyModel, input_data,
                    num_epochs=20, batch_size=128,
                    update_rule='sgd_momentum',
                    optim_config={
                        'learning_rate': 1e-1,
                        },
                    lr_decay = 0.5,
                    verbose=True, 
                    print_every=50)
    t0 = time()
    solver.train()
    solver.Save()
    t1 = time()
    print 'train time %fs' %(t1 - t0)
    
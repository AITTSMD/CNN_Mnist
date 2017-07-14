#coding:utf-8
from classifiers.cnn import *
from datareader.data_utils import *
from layers.layers import *
from layers.layer_utils import *
from solver.solver import *
from time import *
import cPickle
def check_accuracy(X, y, model, num_samples=None, batch_size=100):

   N = X.shape[0]
   if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

   #batch的个数
   num_batches = N / batch_size
   if N % batch_size != 0:
      num_batches += 1
   y_pred = []
   #计算每个batch预测值
   for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
   y_pred = np.hstack(y_pred)
   acc = np.mean(y_pred == y)

   return acc
def check_loss(X, y, model, num_samples=None, batch_size=100):
   sum_loss = 0
   N = X.shape[0]
   if num_samples is not None and N > num_samples:
      mask = np.random.choice(N, num_samples)
      N = num_samples
      X = X[mask]
      y = y[mask]

   #batch的个数
   num_batches = N / batch_size
   if N % batch_size != 0:
      num_batches += 1
   y_pred = []
   #计算每个batch预测值
   for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      loss, _ = model.loss(X[start:end],y[start:end])
      sum_loss = sum_loss+loss

   return sum_loss/num_batches   
if __name__ == "__main__":
   #读取数据
   data = get_MNIST_data()
   f = open('paramdata/model','rb')
   MyModel = cPickle.load(f)
   #print check_accuracy(data['X_test'], data['y_test'],MyModel)
   f.close()
   print check_loss(data['X_test'], data['y_test'],MyModel)
   #print best_params['W1'][0]
   #MyModel.params = best_params
   #y_test_pred = np.argmax(MyModel.loss(data['X_val']),axis=1)
   #print (y_test_pred==data['y_val']).mean()   
   #print best_params.keys()
   #print MyModel.params.keys()
   #for k in MyModel.params.keys():
   #   MyModel.params[k] = best_params[k]   
   
   #print 'load done'
   #print check_accuracy(data['X_train'], data['y_train'],MyModel,100)
   #f.close()
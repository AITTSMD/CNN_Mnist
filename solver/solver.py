#coding:utf-8
import numpy as np
import cPickle
import os
import optim


class Solver(object):
  """
  A Solver encapsulates all the logic necessary for training classification
  models. The Solver performs stochastic gradient descent using different
  update rules defined in optim.py.

  The solver accepts both training and validataion data and labels so it can
  periodically check classification accuracy on both training and validation
  data to watch out for overfitting.

  To train a model, you will first construct a Solver instance, passing the
  model, dataset, and various optoins (learning rate, batch size, etc) to the
  constructor. You will then call the train() method to run the optimization
  procedure and train the model.
  
  After the train() method returns, model.params will contain the parameters
  that performed best on the validation set over the course of training.
  In addition, the instance variable solver.loss_history will contain a list
  of all losses encountered during training and the instance variables
  solver.train_acc_history and solver.val_acc_history will be lists containing
  the accuracies of the model on the training and validation set at each epoch.
  
  Example usage might look something like this:
  
  data = {
    'X_train': # training data
    'y_train': # training labels
    'X_val': # validation data
    'y_val': # validation labels
  }
  model = MyAwesomeModel(hidden_size=100, reg=10)
  solver = Solver(model, data,
                  update_rule='sgd',
                  optim_config={
                    'learning_rate': 1e-3,
                  },
                  lr_decay=0.95,
                  num_epochs=10, batch_size=100,
                  print_every=100)
  solver.train()


  A Solver works on a model object that must conform to the following API:

  - model.params must be a dictionary mapping string parameter names to numpy
    arrays containing parameter values.

  - model.loss(X, y) must be a function that computes training-time loss and
    gradients, and test-time classification scores, with the following inputs
    and outputs:

    Inputs:
    - X: Array giving a minibatch of input data of shape (N, d_1, ..., d_k)
    - y: Array of labels, of shape (N,) giving labels for X where y[i] is the
      label for X[i].

    Returns:
    If y is None, run a test-time forward pass and return:
    - scores: Array of shape (N, C) giving classification scores for X where
      scores[i, c] gives the score of class c for X[i].

    If y is not None, run a training time forward and backward pass and return
    a tuple of:
    - loss: Scalar giving the loss
    - grads: Dictionary with the same keys as self.params mapping parameter
      names to gradients of the loss with respect to those parameters.
  """
  '''
    @model 我们训练好的model
    @data 样本数据
    @kwargs 保存训练model时的一些参数：如update_rule，batch_size等
  '''
  def __init__(self, model, data, **kwargs):
    """
    Construct a new Solver instance.
    
    Required arguments:
    - model: A model object conforming to the API described above
    - data: A dictionary of training and validation data with the following:
      'X_train': Array of shape (N_train, d_1, ..., d_k) giving training images
      'X_val': Array of shape (N_val, d_1, ..., d_k) giving validation images
      'y_train': Array of shape (N_train,) giving labels for training images
      'y_val': Array of shape (N_val,) giving labels for validation images
      
    Optional arguments:
        ���¹���
    - update_rule: A string giving the name of an update rule in optim.py.
      Default is 'sgd'.
        ���¹���������Ҫ�õ��Ĳ��� ��ѧϰ���Ǳ���ģ�     
    - optim_config: A dictionary containing hyperparameters that will be
      passed to the chosen update rule. Each update rule requires different
      hyperparameters (see optim.py) but all update rules require a
      'learning_rate' parameter so that should always be present.
       ѧϰ�ʸ�ʴ��
    - lr_decay: A scalar for learning rate decay; after each epoch the learning
      rate is multiplied by this value.
    minibatches�Ĵ�С
    - batch_size: Size of minibatches used to compute loss and gradient during
      training.
        ���ж�����
    - num_epochs: The number of epochs to run for during training.
        ���ٴε������ӡ
    - print_every: Integer; training losses will be printed every print_every
      iterations.
    - verbose: Boolean; if set to false then no output will be printed during
      training.
    """
    self.model = model
    self.X_train = data['X_train']
    self.y_train = data['y_train']
    self.X_val = data['X_val']
    self.y_val = data['y_val']
    
    #读取训练过程的参数
    #更新规则
    self.update_rule = kwargs.pop('update_rule', 'sgd')
    self.optim_config = kwargs.pop('optim_config', {})
    #学习率的decay
    self.lr_decay = kwargs.pop('lr_decay', 1.0)
    #batch_size大小
    self.batch_size = kwargs.pop('batch_size', 100)
    #训练多少轮
    self.num_epochs = kwargs.pop('num_epochs', 10)

    self.print_every = kwargs.pop('print_every', 10)
    self.verbose = kwargs.pop('verbose', True)

    # Throw an error if there are extra keyword arguments
    if len(kwargs) > 0:
      extra = ', '.join('"%s"' % k for k in kwargs.keys())
      raise ValueError('Unrecognized arguments %s' % extra)
    if not hasattr(optim, self.update_rule):
      raise ValueError('Invalid update_rule "%s"' % self.update_rule)
    self.update_rule = getattr(optim, self.update_rule)

    self._reset()

  #重新reset一些变量
  def _reset(self):
    #轮数
    self.epoch = 0
    #训练过程中最好的准确率
    self.best_val_acc = 0
    #最优参数
    self.best_params = {}
    #历史loss
    self.loss_history = []
    #历史训练准确率
    self.train_acc_history = []
    #历史验证准确率
    self.val_acc_history = []
    self.optim_configs = {}
    for p in self.model.params:
      d = {k: v for k, v in self.optim_config.iteritems()}
      self.optim_configs[p] = d
  #训练一次
  def _step(self):
    num_train = self.X_train.shape[0]
    batch_mask = np.random.choice(num_train, self.batch_size)
    #筛选一个mini-batch
    X_batch = self.X_train[batch_mask]
    y_batch = self.y_train[batch_mask]
    #计算loss和梯度
    loss, grads = self.model.loss(X_batch, y_batch)
    self.loss_history.append(loss)
    #根据优化策略更新参数
    for p, w in self.model.params.iteritems():
      dw = grads[p]
      config = self.optim_configs[p]
      next_w, next_config = self.update_rule(w, dw, config)
      self.model.params[p] = next_w
      self.optim_configs[p] = next_config

  #计算准确率
  def check_accuracy(self, X, y, num_samples=None, batch_size=100):

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
    for i in xrange(num_batches):
      start = i * batch_size
      end = (i + 1) * batch_size
      scores = self.model.loss(X[start:end])
      y_pred.append(np.argmax(scores, axis=1))
    y_pred = np.hstack(y_pred)
    acc = np.mean(y_pred == y)

    return acc


  def train(self):
    
    print 'train11'
    num_train = self.X_train.shape[0]
    #每轮迭代的次数
    iterations_per_epoch = max(num_train / self.batch_size, 1)
    #总的迭代次数
    num_iterations = self.num_epochs * iterations_per_epoch

    for t in xrange(num_iterations):
      self._step()

      if self.verbose and t % self.print_every == 0:
        print '(Iteration %d / %d) loss: %f' % (
               t + 1, num_iterations, self.loss_history[-1])

      epoch_end = (t + 1) % iterations_per_epoch == 0
      #一轮训练完了，更新学习率
      if epoch_end:
        self.epoch += 1
        for k in self.optim_configs:
          self.optim_configs[k]['learning_rate'] *= self.lr_decay
      first_it = (t == 0)
      last_it = (t == num_iterations + 1)
      if first_it or last_it or epoch_end:
        train_acc = self.check_accuracy(self.X_train, self.y_train,
                                        num_samples=1000)
        val_acc = self.check_accuracy(self.X_val, self.y_val)
        #if epoch_end:
        #    if abs(val_acc-self.val_acc_history[-1]<0.0008):
        #        for k in self.optim_configs:
        #            self.optim_configs[k]['learning_rate'] *= 0.75
        self.train_acc_history.append(train_acc)
        self.val_acc_history.append(val_acc)

        if self.verbose:
          print '(Epoch %d / %d) train acc: %f; val_acc: %f' % (
                 self.epoch, self.num_epochs, train_acc, val_acc)
      #保存最好的model
        if val_acc > self.best_val_acc:
          self.best_val_acc = val_acc
          self.best_params = {}
          for k, v in self.model.params.iteritems():
            self.best_params[k] = v.copy()

    self.model.params = self.best_params
  def Save(self):
      loss_str = cPickle.dumps(self.loss_history)
      f = open('paramdata/loss_his.txt','wb')
      f.write(loss_str)
      f.close()
      train_acc_str = cPickle.dumps(self.train_acc_history)
      f = open('paramdata/train_acc_his.txt','wb')
      f.write(train_acc_str)
      f.close()
      val_acc_str = cPickle.dumps(self.val_acc_history)
      f = open('paramdata/val_acc_his.txt','wb')
      f.write(val_acc_str)
      f.close()
      best_params_str = cPickle.dumps(self.best_params)
      f = open('paramdata/best_params.txt','wb')
      f.write(best_params_str)
      f.close()
      #保存最好的模型
      f = open('paramdata/model','wb')
      cPickle.dump(self.model,f,protocol=cPickle.HIGHEST_PROTOCOL)
      f.close()
      print 'done'
      
      
      
      
      
      
      
# CNN_Mnist
本项目结合cs231n的课后作业，使用CNN完成手写字符的识别，已独立实现卷积、池化、BN、dropout操作，但训练过程太过缓慢，所以实际的训练过程中卷积、池化操作参考了cs231n提供的代码。 最终在测试集上的准确率为97.43%，train error为0.071207 test error为0.0959。本项目旨在学习模型的搭建和BP算法。希望对初学者有所帮助！

## 注意
 1. 在layers/layers.py中conv_forward_naive和max_pool_forward_naive均为慢速版本的卷积 和池化操作，训练速度很慢！
 2. layers/fast_layers.py中定义了卷积和池化的快速版本，使用前需要到根目录中执行
 **python setup.py build_ext --inplace**命令。
 3. 快速版本的最大池化都是**non-overlapping**，即**移动步长要不能小于池化窗口的大小**
## 开发环境
 * Ubuntu 16.04
 * python 2.7
 * jupyter notebook
## 网络配置
采用了较为简单的卷积神经网络模型，包含4个卷积层，2个池化层和2个全连接层，读者可以根据layers/layers.py中定义的各种类型的层进行模型的搭建。为了加快模型收敛，在卷积层、全连接层中均使用了Batch Normalization。网络模型如下图所示：



![网络模型](https://i.loli.net/2017/07/14/596869d4c7d0a.png)

## 实验结果
![train_loss.png](https://i.loli.net/2017/07/14/59686a5993c04.png)
![train_val_accuracy.png](https://i.loli.net/2017/07/14/59686a83ab80a.png)




# z_bp

基于Numpy动手实现一个dnn训练框架:z_dnn，支持全连接层的正反传播，支持Sigmoid和Tanh激活

通过把机器学习的理论、原理和工程实践结合起来，在动手“建造”中加深对理论的“理解”。神经框架正是一个最好的例子。但是，以pytorch为代表的现代深度学习框架的源码对于学习而言已经过于复杂，因此，我决定用纯Python+Numpy实现一个简单的bp框架。
我在代码中写了尽量详细的注释，通过手动实现正向预测反向传播，并从中学习和理解机器学习背后的原理。

著名物理学家，诺贝尔奖得主Richard Feynman办公室的黑板上写了："What I cannot create, I do not understand."。

特性

- 实现了自定义超参数（模型深度，结构，激活函数，学习率，banth大小）。
- 实现了基本的完数据集据导入，训练，验证，预测的过程

## 以MNIST数据集为例

> 训练了600轮后的效果

![6302350316f2c2beb17fb0d8](https://user-images.githubusercontent.com/69743646/185793856-3546f997-54bd-4e53-8dc8-6717b4401567.png)


> 测试精度

![12356](https://user-images.githubusercontent.com/69743646/185793811-38add978-f429-4107-aec2-1b21449bbde0.PNG)




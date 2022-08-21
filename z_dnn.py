import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import random


def tanh(x):
    return np.tanh(x)

def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()
def bypass(x):
    return  x;

def d_tanh(data):
    return 1 / (np.cosh(data)) ** 2
    #return np.diag(1/(np.cosh(data))**2)

def d_softmax(data):
    sm = softmax(data)
        # diag:对角矩阵  outer：第一个参数挨个乘以第二个参数得到矩阵
    return np.diag(sm) - np.outer(sm, sm)
def d_bypass(x):
    return 1;

weight_scale = 1e-3


differential = {softmax: d_softmax, tanh: d_tanh, bypass:d_bypass}

d_type={softmax:1, tanh:0}

class Z_dnn (object):

    # 一、超参数
    # 层数和节点数
    dimensions = []
    # 激活函数
    act=[]
    #数据集
    #data=[]
    train_x = []
    train_y = []
    valid_x = []
    valid_y = []
    test_x = []
    test_y = []

    #训练参数
    batch_size=100
    lr=1.0
    #模型
    para=[]

    #中间数据
    l_in_list=[]
    l_out_list=[]


    def __init__(self,dimensions,act):
        self.dimensions=dimensions
        self.act=act
        self.trueY = np.identity(self.dimensions[-1])




        #模型初始化
    def init_para(self):
        self.para=[{}]
        for layer in range(1,len(self.dimensions)):
            layer_parameter = {}
            layer_parameter['w'] = weight_scale * np.random.randn(self.dimensions[layer-1],self.dimensions[layer])
            layer_parameter['b'] = np.zeros(self.dimensions[layer])
            self.para.append(layer_parameter)
        return self.para


    #存入数据集
    def init_data(self,train_x = [],train_y = [],valid_x = [],valid_y = [], test_x = [],test_y = []):
        if len(train_x):
            self.train_x = train_x.reshape(-1, self.dimensions[0])
            self.train_y = train_y.reshape(-1, self.dimensions[-1])
            self.train_num=len(train_x)
        if len(valid_x):
            self.valid_x = valid_x.reshape(-1, self.dimensions[0])
            self.valid_y = valid_y.reshape(-1, self.dimensions[-1])
        if len(test_x):
            self.test_x = test_x.reshape(-1, self.dimensions[0])
            self.test_y = test_y.reshape(-1, self.dimensions[-1])

    #训练参数
    def init_train(self,batch_size=100,lr=1.0):
        self.batch_size=batch_size
        self.lr=lr

    # 预测函数
    def predict(self,val):
        l_out=val
        for layer in range(1,len(self.dimensions)):
            l_in=np.dot(l_out, self.para[layer]['w']) + self.para[layer]['b']
            l_out=self.act[layer](l_in)
        return l_out

    #梯度下降
    def grad_para(self,x,y):
        l_in_list=[0]
        l_out_list=[x]
        for layer in range(1, len(self.dimensions)):
            l_in = np.dot(l_out_list[layer-1], self.para[layer]['w']) + self.para[layer]['b']
            l_out = self.act[layer](l_in)
            l_in_list.append(l_in)
            l_out_list.append(l_out)

        d_layer=-2*(self.trueY[0][y]-l_out_list[-1])
        grad=[None]*len(self.dimensions)
        for layer in range(len(self.dimensions)-1,0,-1):
            if self.act[layer]==tanh:
                d_layer=differential[self.act[layer]](l_in_list[layer])*d_layer
            else :  d_layer=np.dot(differential[self.act[layer]](l_in_list[layer]),d_layer)
            grad[layer]={}
            grad[layer]['b']=d_layer
            grad[layer]['w']=np.outer(l_out_list[layer-1],d_layer)
            d_layer=np.dot(self.para[layer]['w'],d_layer)

        return grad

    # 一次求若干张图片的梯度，取均值作为下降方向
    def train_grad(self,num=0):
        b=num * self.batch_size
        grad_acc = self.grad_para(self.train_x[b],self.train_y[b])
        for img_i in range(1, self.batch_size):
            grad_n = self.grad_para(self.train_x[b+img_i],self.train_y[b+img_i])
            for layer in range(1,len(self.dimensions)):
                grad_acc[layer]['w'] += grad_n[layer]['w']
                grad_acc[layer]['b'] += grad_n[layer]['b']

        for layer in range(1, len(self.dimensions)):
            grad_acc[layer]['w'] /= self.batch_size
            grad_acc[layer]['b'] /= self.batch_size
        return grad_acc

    # 训练过程
    def learn(self):
        for i in range(self.train_num // self.batch_size):
            if i % 10 == 0:
                print("running learn {}/{}".format(i + 10, self.train_num // self.batch_size),self.valid_accuracy())
                self.valid_accuracy()
            grad = self.train_grad(i)

            for layer in range(1, len(self.dimensions)):
                self.para[layer]['w'] -= self.lr * grad[layer]['w']
                self.para[layer]['b'] -= self.lr * grad[layer]['b']

    #测试精度
    def valid_accuracy(self):
        correct = [self.predict(self.valid_x[i]).argmax() is self.valid_y[i] for i in range(len(self.valid_y))]
        return correct.count(True) / len(correct)



        # 保存模型
    def saveMod(self, filename):
        f = open(filename, 'wb')
        pickle.dump(self.para, f)
        f.close()
        return filename

    # 加载模型
    def loadMod(self, filename):
        f = open(filename, 'rb')
        self.para = pickle.load(f)
        f.close()


#新建网络对象，初始化参数和激活函数
myM=Z_dnn([28*28,256,10],[0,tanh,softmax])
#初始化模型
myM.init_para()
#导入模型
myM.loadMod('mode_1.txt')
#导出模型
myM.saveMod('mode_1.txt')
#加载数据集
data = np.load("mnist.npz")
#分割训练集和验证集
myM.init_data(data['x_train'][:50000],data['y_train'][:50000],data['x_train'][50000:60000],data['y_train'][50000:60000])
#设置训练参数，包括batch_size和学习率
myM.init_train(100,0.5)
#开始训练
myM.learn()
#验证当前模型精度
myM.valid_accuracy()
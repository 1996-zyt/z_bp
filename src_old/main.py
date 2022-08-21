import numpy as np
import math
import matplotlib.pyplot as plt
import pickle
import random



# 保存模型
def saveMod(v, filename):
    f = open(filename, 'wb')
    pickle.dump(v, f)
    f.close()
    return filename


# 加载模型
def loadMod(filename):
    f = open(filename, 'rb')
    r = pickle.load(f)
    f.close()
    return r


##filename = save_variable(results,'results.txt')
##results = load_variavle('results.txt')

# 加载数据集
data = np.load("mnist.npz")
train_img = data['x_train'][:50000].reshape(-1, 28 * 28)
train_lab = data['y_train'][:50000]
valid_img = data['x_train'][50000:60000].reshape(-1, 28 * 28)
valid_lab = data['y_train'][50000:60000]
test_img = data['x_test'].reshape(-1, 28 * 28)
test_lab = data['y_test']
train_list=list(range(50000))
random.shuffle(train_list)
train_num = 50000
valid_num = 10000
test_num = 10000
pace = 100


# 显示图片
def show(img):
    plt.imshow(img.reshape(28, 28), cmap='gray')
    plt.show()


# 0层激活函数
def tanh(x):
    return np.tanh(x)


# tanh导数函数
# def d_tanh(data):
# 	return np.diag(1/(np.cosh(data))**2)
# tanh导数函数优化：
def d_tanh(data):
    return 1 / (np.cosh(data)) ** 2


# 1层激活函数
def softmax(x):
    exp = np.exp(x - x.max())
    return exp / exp.sum()


# softmax导数函数
def d_softmax(data):
    sm = softmax(data)
    # diag:对角矩阵  outer：第一个参数挨个乘以第二个参数得到矩阵
    return np.diag(sm) - np.outer(sm, sm)


# 初始化模型
def init_para():
    para = {}
    d = math.sqrt(6 / (dimen[0] + dimen[1]))
    para['b0'] = np.random.rand(dimen[0]) * 0
    para['w1'] = np.random.rand(dimen[0], dimen[1]) * 2 * d - d
    para['b1'] = np.random.rand(dimen[1]) * 0
    return para





# 按给定模型预测结果预测
def predict(img, para):
    l0_in = img + para['b0']
    l0_out = activation['A0'](l0_in)
    l1_in = np.dot(l0_out, para['w1']) + para['b1']
    l1_out = activation['A1'](l1_in)
    return l1_out


# 预测值与实际的方差
def sqr_loss(img, lab, para):
    y_pred = predict(img, para)
    y = trueY[lab]
    diff = y - y_pred
    return np.dot(diff, diff)


# 计算单步梯度（算法核心）
def grad(img, lab, para):
    l0_in = img + para['b0']
    l0_out = activation['A0'](l0_in)
    l1_in = np.dot(l0_out, para['w1']) + para['b1']
    l1_out = activation['A1'](l1_in)
    diff = trueY[lab] - l1_out
    act = np.dot(activation['d_A1'](l1_in), diff)
    grad_b1 = -2 * act
    grad_w1 = -2 * np.outer(l0_out, act)
    grad_b0 = -2 * activation['d_A0'](l0_in) * np.dot(para['w1'], act)
    return {'b1': grad_b1, 'w1': grad_w1, 'b0': grad_b0}


# 一次求100张图片的梯度，取均值作为下降方向
def train_grad(num, para):
    grad_acc = grad(train_img[train_list[num * pace + 0]],
                    train_lab[train_list[num * pace + 0]], para)
    for img_i in range(1, pace):
        grad_n = grad(train_img[train_list[num * pace + img_i]],
                      train_lab[train_list[num * pace + img_i]], para)
        for key in grad_acc.keys():
            grad_acc[key] += grad_n[key]
    for key in grad_acc.keys():
        grad_acc[key] /= pace
    return grad_acc


# 训练过程
def learn(rate, para):
    for i in range(train_num // pace):
        if i % 10 == 0:
            print("running learn {}/{}".format(i+10, train_num // pace))
            valid_accuracy(para)
        grad = train_grad(i, para)
        para['b0'] -= rate * grad['b0']
        para['b1'] -= rate * grad['b1']
        para['w1'] -= rate * grad['w1']
    return para




# 再验证集上的精度
def valid_accuracy(para):
    correct = [predict(valid_img[img_i], para).argmax() == valid_lab[img_i] for img_i in range(valid_num)]
    print("valid_accuracy:{}".format(correct.count(True) / len(correct)))

def test_accuracy(para):
    correct=[0]*10000
    w=[0]*10
    for img_i in range(test_num):
        if predict(test_img[img_i], para).argmax() == test_lab[img_i]:
            correct[img_i]=True
        else :
            correct[img_i]=False
            w[test_lab[img_i]]+=1

    #correct = [predict(test_img[img_i], para).argmax() == test_lab[img_i] for img_i in range(test_num)]
    print("test_accuracy:{}".format(correct.count(True) / len(correct)))
    print(w)

# 创建bp神经网络
dimen = [28 * 28, 10]  # 维度
trueY = np.identity(dimen[-1])
activation = {'A0': tanh, 'd_A0': d_tanh, 'A1': softmax, 'd_A1': d_softmax, }


#近行模型训练
#para = init_para()
para = loadMod('mode_0.txt')
para1 = loadMod('mode_0.txt')
#rate=0.5
#for i in range(10):
#    random.shuffle(train_list)
#    mode = learn(rate, para)
#    rate /= 2

test_accuracy(para)
#
##saveMod(mode,'mode_0.txt')

#a=np.transpose(para1['w1'])
#for i in range(10):
#    plt.imshow(a.reshape(-1,28, 28)[i], cmap='gray')
#    print(predict(a[i],para).argmax())
#    plt.show()





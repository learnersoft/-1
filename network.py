import numpy as np
import scipy.special
import matplotlib.pyplot as plt
import struct
class neuralNetwork :

    # 用于神经网络初始化
    def __init__(self, inputnodes, hiddennodes, outputnodes, learningrate,alpha):
        # 输入层节点数
        self.inodes = inputnodes
        # 隐层节点数
        self.hnodes = hiddennodes
        # 输出层节点数
        self.onodes = outputnodes
        # 学习率
        self.lr = learningrate

        # 初始化输入层与隐层之间的权重
        self.wih = np.random.normal(0.0, pow(self.hnodes, -0.5), (self.hnodes, self.inodes))
        # 初始化隐层与输出层之间的权重
        self.who = np.random.normal(0.0, pow(self.onodes, -0.5), (self.onodes, self.hnodes))
        self.bih = np.random.normal(0.0, pow(self.onodes, -0.5))
        self.bho = np.random.normal(0.0, pow(self.onodes, -0.5))
        self.activation_function = lambda x: scipy.special.expit(x)
        self.alpha=alpha
        self.loss=0
    # 神经网络学习训练
    def train(self, inputs_list, targets_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T
        # 将输入标签转化成二维矩阵
        targets = np.array(targets_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)+self.bih
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)


        final_inputs = np.dot(self.who, hidden_outputs)+self.bho
        final_outputs = np.exp(final_inputs) / np.sum(np.exp(final_inputs))
        self.loss=-np.sum(targets*np.log(final_outputs+0.0001))
        # 计算输出层误差
        output_errors = final_inputs-targets

        # 计算隐层误差
        hidden_errors = hidden_outputs*(1-hidden_outputs)*np.dot(self.who.T, output_errors)
        self.who-=self.lr*((output_errors.dot(hidden_outputs.T))+self.alpha*self.who)
        self.wih-=self.lr*(hidden_errors.dot(inputs.T)+self.alpha*self.wih)
        self.bho-=self.lr*(output_errors+self.alpha*self.bho)
        self.bih-=self.lr*(hidden_errors+self.alpha*self.bih)
        return self.loss
    def test(self, inputs_list):
        # 将输入数据转化成二维矩阵
        inputs = np.array(inputs_list, ndmin=2).T

        # 计算隐层的输入
        hidden_inputs = np.dot(self.wih, inputs)
        # 计算隐层的输出
        hidden_outputs = self.activation_function(hidden_inputs)

        # 计算输出层的输入
        final_inputs = np.dot(self.who, hidden_outputs)
        # 计算输出层的输出
        final_outputs = np.exp(final_inputs) / np.sum(np.exp(final_inputs))

        return final_outputs
    def learning_rate_decay(self):
        self.lr*=0.995
def data_fetch_preprocessing():
    train_image = open('train-images.idx3-ubyte', 'rb')
    test_image = open('t10k-images.idx3-ubyte', 'rb')
    train_label = open('train-labels.idx1-ubyte', 'rb')
    test_label = open('t10k-labels.idx1-ubyte', 'rb')

    magic, n = struct.unpack('>II',
                             train_label.read(8))
    # 原始数据的标签
    y_train_label = np.array(np.fromfile(train_label,
                                         dtype=np.uint8), ndmin=1)
    y_train = np.ones((10, 60000)) * 0.01
    for i in range(60000):
        y_train[y_train_label[i]][i] = 0.99

    # 测试数据的标签
    magic_t, n_t = struct.unpack('>II',
                                 test_label.read(8))
    y_test = np.fromfile(test_label,
                         dtype=np.uint8).reshape(10000, 1)
    # print(y_train[0])
    # 训练数据共有60000个
    # print(len(labels))
    magic, num, rows, cols = struct.unpack('>IIII', train_image.read(16))
    x_train = np.fromfile(train_image, dtype=np.uint8).reshape(len(y_train_label), 784).T

    magic_2, num_2, rows_2, cols_2 = struct.unpack('>IIII', test_image.read(16))
    x_test = np.fromfile(test_image, dtype=np.uint8).reshape(len(y_test), 784).T
    # print(x_train.shape)
    # 可以通过这个函数观察图像
    # data=x_train[:,0].reshape(28,28)
    # plt.imshow(data,cmap='Greys',interpolation=None)
    # plt.show()
    x_train = x_train / 255 * 0.99 + 0.01
    x_test = x_test / 255 * 0.99 + 0.01

    # 关闭打开的文件
    train_image.close()
    train_label.close()
    test_image.close()
    test_label.close()

    return x_train, y_train, x_test, y_test
if __name__ == "__main__":
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    x_train, y_train, x_test, y_test=data_fetch_preprocessing()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10


    learning_rate = 0.01
    alpha=0.001
    # 训练次数
    epochs = 10
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,alpha)

    random_list=np.arange(x_train.shape[1])
    loss_list=[]
    for e in range(epochs):
        np.random.shuffle(random_list)
        print('iter',e)
        for i in random_list:
            if i%100==0:
                n.learning_rate_decay()
            inputs=x_train[:,i]
            targets=y_train[:,i]
            loss=n.train(inputs, targets)
            loss_list.append(loss)
    plt.plot(range(len(loss_list)),loss_list)
    plt.xlabel('iter')
    plt.ylabel('loss')
    plt.title('train loss')
    np.save('loss.npy',np.array(loss_list))
    acc=0
    for i in range(x_test.shape[1]):
        results = n.test(x_test[:,i])
        if results.argmax()==y_test[i][0]:
            acc+=1
    acc=acc/x_test.shape[1]
    print('测试',acc)
    np.save('w1.npy',n.who)
    np.save('w2.npy',n.wih)

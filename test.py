from network import neuralNetwork,data_fetch_preprocessing
import numpy as np
if __name__ == "__main__":
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    x_train, y_train, x_test, y_test=data_fetch_preprocessing()

    input_nodes = 784
    hidden_nodes = 200
    output_nodes = 10


    learning_rate = 0.01
    alpha=0.001
    # 训练次数
    epochs = 10
    # 初始化神经网络实例
    n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate,alpha)
    n.who=np.load("w1.npy")
    n.wih=np.load("w2.npy")

    acc=0
    for i in range(x_test.shape[1]):
        results = n.test(x_test[:,i])
        if results.argmax()==y_test[i][0]:
            acc+=1
    acc=acc/x_test.shape[1]
    print('测试',acc)
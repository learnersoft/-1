from network import neuralNetwork,data_fetch_preprocessing
import numpy as np
if __name__ == "__main__":
    hidden_nodes_list=[20,50,200]
    learning_rate_list=[0.01,0.1,1]
    alpha_list=[0.001,0.01,0.1]
    # 初始化 784（28 * 28）个输入节点，100个隐层节点，10个输出节点（0~9）
    input_nodes = 784
    output_nodes = 10

    # 训练次数
    epochs = 10
    x_train, y_train, x_test, y_test = data_fetch_preprocessing()
    print(x_train.shape,y_train.shape,x_test.shape,y_test.shape)
    x_val = x_train[:,int(x_train.shape[1] * 0.8):]
    x_train = x_train[:,:int(x_train.shape[1] * 0.8)]
    y_val = y_train[:,int(y_train.shape[1] * 0.8):]
    y_train = y_train[:,:int(y_train.shape[1] * 0.8)]
    print(x_train.shape,x_val.shape,y_train.shape,y_val.shape)
    for learning_rate in learning_rate_list:
        for hidden_nodes in hidden_nodes_list:
            for alpha in alpha_list:
                # 初始化神经网络实例
                n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate, alpha)
                random_list = np.arange(x_train.shape[1])
                for e in range(epochs):
                    np.random.shuffle(random_list)
                    for i in random_list:
                        if i%100==0:
                            n.learning_rate_decay()
                        inputs = x_train[:, i]
                        targets = y_train[:, i]
                        n.train(inputs, targets)
                acc=0
                for i in range(x_val.shape[1]):
                    results = n.test(x_val[:,i])
                    if results.argmax() == y_val[:,i].argmax():
                        acc += 1
                acc = acc / x_val.shape[1]
                print('测试', acc)
                print('leaning_rate=',learning_rate,'hidden_nodes=',hidden_nodes,'alpha=',alpha,"acc=",acc)
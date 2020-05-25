import torch as th
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import utils
import matplotlib.pyplot as plt


class Net(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Net, self).__init__()
        self.liner = nn.Linear(in_features=input_dim, out_features=1)

    def forward(self, x):
        x = self.liner(x)
        return x




def LRModel():
    print("==============Linear Regression Model training=====================")
    def draw_pred_real_graph(train_x,test_x,train_y,test_y,m:Net):
        all_x = np.vstack([train_x,test_x])
        all_y = np.vstack([train_y,test_y])
        all_x = th.from_numpy(all_x)
        pre = m(all_x)
        plt.scatter(range(len(all_x[0:50])),pre.detach().numpy()[0:50],marker='x',c='red',label='Predict')
        plt.plot(range(50),all_y[0:50],Linewidth=1,c='orange',Linestyle='-',label = 'real')
        plt.legend()
        plt.title('LR Model Predict and Real')
        plt.savefig('data\\LR Model Predict and Real.png')
        plt.show()

    train_x, train_y, test_x, test_y =  utils.data_process()
    model = Net(254, 1)
    criterion = th.nn.MSELoss()    # Defined loss function
    optimizer = th.optim.SGD(model.parameters(), lr=0.01,weight_decay=1e-2)


    train_x = th.from_numpy(train_x).float()
    train_x = Variable(train_x)
    train_y = th.from_numpy(train_y).float()
    train_y = Variable(train_y.reshape(-1, 1))
    test_x = th.from_numpy(test_x).float()
    test_x = Variable(test_x)
    test_y = th.from_numpy(test_y.reshape(-1, 1)).float()
    test_y = Variable(test_y)

    # 模型训练
    train_loss = []
    test_loss = []
    for epoch in range(10000):
        # Forward pass
        y_pred = model(train_x)
        # Compute loss
        loss = criterion(y_pred, train_y)
        t = criterion(model(test_x),test_y)
        if (epoch+1) % 100 == 0:
            print("epoch:", epoch+1, "RMSE", np.sqrt(loss.item()), "MSE", loss.item())
        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()
        train_loss.append(np.sqrt(loss.item()))
        test_loss.append(np.sqrt(t.item()))
    print("=============LinerRegressionModel model test==================")
    model.eval()
    out = model(test_x)
    loss = criterion(out, test_y)
    print("test loss:", np.sqrt(loss.item()))
    th.save(model, 'LinerRegressionModel.pth')
    """========================"""
    draw_pred_real_graph(train_x,test_x,train_y,test_y,model)
    return test_loss,train_loss,np.sqrt(loss.item())




if __name__ == '__main__':
    """
    res = []
    for i in range(100):
        test_, loss_ = LRModel()
        res.append(test_[-1])
    print(sum(res)/len(res))
    """
    avg = 1
    for i in range(1):
        res = []
        test_, loss_,rmse_test= LRModel()
        print("iter:",i)
        res.append(rmse_test)
    print("average RMSE:",sum(res)/len(res))
    plt.plot(test_, label='test')
    plt.plot(loss_, label='train')
    plt.xlabel("iteration")
    plt.ylabel("RMSE LOSS")
    plt.title("Linear Regression Model")
    plt.legend()
    plt.savefig('data\\LR_Model_train_Test_loss.png')
    plt.show()

import torch as th
import pandas as pd
import numpy as np
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import  utils
import matplotlib.pyplot as plt





class Net(nn.Module):
    def __init__(self, input_dim, output_dim,hidensize):
        super(Net, self).__init__()
        self.linear = nn.Linear(in_features=input_dim, out_features=hidensize,bias=True)
        self.layer_2 = nn.Linear(in_features=hidensize, out_features=1,bias=True)

        self.DropOut = nn.Dropout(0.5)

    def forward(self, x):
        x = self.linear(x)
        x = th.sigmoid(x)

        #x = self.DropOut(x)

        x = self.layer_2(x)
        #x = self.DropOut(x)


        return x


def draw_pred_real_graph(test_x,test_y,m,fig_name):

    pre = m(test_x)
    plt.scatter(range(len(pre[0:50])), pre.detach().numpy()[0:50], marker='x',c='red',label='Predict')
    plt.plot(test_y.detach().numpy()[0:50], Linewidth=1, c='orange', Linestyle='-', label='real')
    plt.legend()
    plt.title(fig_name)
    plt.savefig('data\\'+fig_name)
    plt.show()


def PerceptronModel(hiden_size):
    data = utils.load_data()
    data = utils.normalization(data,normal_y=True)
    data = utils.process_missing_data(data)
    data = utils.one_hot(data)
    train_x, train_y, test_x, test_y = utils.train_test_sp(data, 0.3)

    model = Net(254, 1,hiden_size)

    criterion = th.nn.MSELoss()    # Defined loss function
    optimizer = th.optim.Adam(model.parameters(), lr=0.003,weight_decay=1e-5)


    train_x = th.from_numpy(train_x).float()
    train_x = Variable(train_x)
    train_y = th.from_numpy(train_y).float()
    train_y = Variable(train_y.reshape(-1, 1))
    test_x = th.from_numpy(test_x).float()
    test_x = Variable(test_x)
    test_y = th.from_numpy(test_y.reshape(-1, 1)).float()
    test_y = Variable(test_y)

    # 模型训练
    loss_ = []
    t_loss = []
    for epoch in range(2000):
        # Forward pass
        y_pred = model(train_x)

        # Compute loss

        loss = criterion(y_pred, train_y)
        loss_.append(loss.item())
        t_y = model(test_x)
        t_l = criterion(t_y,test_y)
        t_loss.append(t_l.item())

        if (epoch+1) % 100 == 0:
            print("epoch:", epoch+1, "RMSE", np.sqrt(loss.item()), "MSE", loss.item())

        # Zero gradients
        optimizer.zero_grad()
        # perform backward pass
        loss.backward()
        # update weights
        optimizer.step()

    draw_pred_real_graph(test_x,test_y,m=model,fig_name='Perceptron Predict and real')

    print("=============Perceptron model test==================")
    model.eval()
    out = model(test_x)
    loss = criterion(out, test_y)
    print("test loss:", np.sqrt(loss.item()))
    th.save(model, 'PerceptronModel.pth')
    plt.plot(t_loss,label='test')
    plt.plot(loss_,label='train')
    plt.title("Perceotron Model MSE Loss")
    plt.legend()
    plt.savefig("data\\Perceotron Model MSE Loss")
    plt.show()

    return np.sqrt(loss.item())


if __name__ == '__main__':

    res = []
    for i in range(100):
        loss = PerceptronModel(384)
        res.append(loss)
  
    print('average:',sum(res)/len(res))

    PerceptronModel(384)
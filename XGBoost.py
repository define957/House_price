from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import ShuffleSplit
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import xgboost as xgb
from sklearn.metrics import mean_squared_error
import  utils
import graphviz



def xgboost_model():
    data = utils.load_data()
    data = utils.normalization(data,normal_y=True)
    data = utils.process_missing_data(data)
    data = utils.one_hot(data)
    train_x, train_y, test_x, test_y = utils.train_test_sp(data,0.3)
    matrix=xgb.DMatrix(data=train_x,label=train_y)
    xg_reg=xgb.XGBRegressor(objective='reg:linear',colsample_bytree=0.9,learning_rate=0.01,max_depth=7,alpha=2,n_estimators=1400,subsample=0.95,reg_alpha=0.65,reg_lambda=0.45)
    xg_reg.fit(train_x,train_y)
    pred = xg_reg.predict(test_x)
    rmse = np.sqrt(mean_squared_error(test_y, pred))
    print("=============xgboost model test==================")
    print("test loss",rmse)


    plt.scatter(range(len(pred[0:50])),pred[0:50],marker='x',c='red',label='predict')
    plt.plot(test_y[0:50],label='test target')
    plt.title("XGBoost Predict and real")
    plt.legend()
    plt.savefig("data\\XGBoost Predict and real.png")
    plt.show()

    return rmse,xg_reg,pred,test_y


if __name__ == '__main__':
    """
    res = []
    for i in range(100):
        RMSE ,_ = xgboost_model()
        res.append(RMSE)
    print("mean RMSE:", sum(res)/len(res))
    """
    RMSE,model,pred,test_y = xgboost_model()
    print("RMSE:",RMSE)

    xgb.plot_importance(model,max_num_features=10)
    plt.show()


    """
    digraph = xgb.to_graphviz(model, num_trees=200)
    digraph.format = 'png'
    digraph.view('./t1')

    digraph = xgb.to_graphviz(model, num_trees=500)
    digraph.format = 'png'
    digraph.view('./t2')
    """

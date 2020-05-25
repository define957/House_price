import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import tree
from sklearn.feature_selection import SelectFromModel
from scipy import stats as stats


def check_null(data):
    """
    查看数据集缺失情况
    :param data: 原始数据集
    :return: 数据集缺失
    """
    null_count = data.isnull().sum()
    null_count = null_count.sort_values(ascending=False)
    missing_rate = null_count / 1469
    null_matrix = pd.concat([null_count, missing_rate], axis=1, keys=['count', 'rate'])
    return null_matrix

def plot_loss_data(null_matrix:pd.DataFrame,fig_name:str):
    """
    绘制数据缺失情况图
    :param null_matrix: check_null函数的返回值
    :param fig_name: 图像名称
    :return:None
    """
    plt.figure(figsize=(10,8))
    loss_rate = null_matrix.iloc[0:20, 1]
    index = null_matrix.head(20).index.values
    x = np.arange(0,20)
    plt.xticks(x,index,rotation=45)
    plt.plot(x,loss_rate.values,linewidth=2.5,c='red')
    plt.title("features missing rate")
    plt.savefig('data\\'+fig_name)
    plt.show()

def data_corr(data:pd.DataFrame,fig_name:str):
    """
    返回相关系数矩阵
    :param data:训练样本
    :param fig_name:图像名称
    :return:None
    """
    plt.figure(figsize=(10,8))
    corr_matrix = data.corr()
    sns.heatmap(corr_matrix,cmap='YlGnBu')
    plt.savefig("data\\"+fig_name)
    plt.show()


def normal_and_original_graph():
    """
    绘制销售价格的频率直方图与该样本对应正态分布图像
    :return:
    """
    bins = 30
    (mu, sigma) = stats.norm.fit(data['SalePrice'])
    y = stats.norm.pdf(range(800000), mu, sigma)
    plt.plot(y)
    sns.distplot(data['SalePrice'].values, norm_hist=1, color="orange", bins=bins, kde=True)  # kde显示拟合曲线
    plt.legend(["mu={},sigma={}".format(mu,sigma)], loc='best')
    plt.savefig("data\\频率分布直方图.png")
    plt.show()

def log_graph():
    bins = 30
    logdata = np.log1p(data['SalePrice'].values) #响应变量对数化
    (mu, sigma) = stats.norm.fit(logdata)
    x = np.linspace(start=8, stop=14, num=50)
    y = stats.norm.pdf(x, mu, sigma)
    plt.plot(x, y)
    sns.distplot(np.log1p(data['SalePrice'].values), norm_hist=1, color="orange", bins=bins, kde=True)  # kde显示拟合曲线
    plt.legend(["mu={},sigma={}".format(mu, sigma)], loc='best')
    plt.savefig("data\\对数化后频率直方图.png")
    plt.show()
    





data = pd.read_csv('data\\train.csv')
null_matrix = check_null(data)
plot_loss_data(null_matrix,"features_missing_rate")
data_corr(data,'数字变量相关性图像')
normal_and_original_graph()
fig = plt.figure()
res = stats.probplot(data['SalePrice'], plot=plt)
plt.show()
log_graph()
data['PoolQC'] = data['PoolQC'].apply(lambda x: 'no_pool' if pd.isnull(x) else x)
data = data.drop('MiscFeature',axis=1)
numeric_data = data.dtypes[data.dtypes != 'object'].index
data[numeric_data] = data[numeric_data].apply(lambda x:(x-x.mean())/(x.std()))
data = pd.get_dummies(data, dummy_na=True)

from XGBoost import xgboost_model
from PerceptronModel import PerceptronModel
from LinerRegressionModel import LRModel


if __name__ == '__main__':
    """线性回归模型"""
    LRModel()
    """多重感知机模型"""
    PerceptronModel(384)
    """XGBoost模型"""
    xgboost_model()





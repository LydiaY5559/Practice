# coding:utf-8
import itertools

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --------------------------------------------------------------------------------
# preprocessing
# --------------------------------------------------------------------------------
def preprocessing_minmax(X, axis=0):
    X = np.array(X)
    X_min = X.min(axis=axis, keepdims=True)
    X_max = X.max(axis=axis, keepdims=True)
    return (X - X_min) / (X_max - X_min)


# --------------------------------------------------------------------------------
# plot
# --------------------------------------------------------------------------------
# 绘制分类器预测分布图（仅限特征维度为2）
def plot_decision_boundary(X, y, pred_func, h=0.02):
    # Set min and max values and give it some padding
    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    # Generate a grid of points with distance h between them
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    # Predict the function value for the whole gid
    Z = pred_func(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    # Plot the contour and training examples
    plt.contourf(xx, yy, Z, cmap=plt.cm.Spectral, alpha=0.9)
    plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Spectral)

# 绘制分类器在不同参数下的预测分布图
def plot_classifier_paras(model, X, y, paras_dict, plot_input=True, col_num=4):
    # plot设置
    figsize_col = 15
    figsize_row = figsize_col / col_num * 0.85
    
    # 转换paras格式 {'a': [1,2], 'b': [10,20]} -> [{'a':1, 'b':10}, ...] 即生成笛卡尔积
    paras_keys = list(paras_dict.keys())
    paras_values = itertools.product(*paras_dict.values())
    paras_list = [dict(zip(paras_keys, values)) for values in paras_values]
    
    if plot_input: paras_list = np.append([None], paras_list)
    plot_row_num = (len(paras_list) - 1) // col_num + 1
    plt.figure(figsize=(figsize_col, plot_row_num * figsize_row))
    
    for i, paras in enumerate(paras_list):
        plt.subplot(plot_row_num, col_num, i+1)
        if plot_input and i == 0:
            plt.title('input')
            plt.scatter(X[:,0], X[:,1], c=y, cmap=plt.cm.Spectral)
            continue
        plt.title(', '.join(['%s = %s'%(k,v) for k,v in paras.items()]))
        model_obj = model(**paras)
        model_obj.fit(X, y)
        plot_decision_boundary(X, y, model_obj.predict)
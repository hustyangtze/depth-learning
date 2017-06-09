# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 13:34:39 2017

@author: XIE
"""

import cPickle
import gzip
import numpy as np

#从数据集中载入数据
def load_data():
    f=gzip.open('../data/mnist.pkl.gz','rb')
    training_data,validation_data,test_data=cPickle.load(f)
    f.close()
    return (training_data, validation_data, test_data)

#改编数据集的格式
def load_data_wrapper():
    tr_d,va_d,te_d = load_data()
    #训练集
    training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
    training_results = [vectorized_result(y) for y in tr_d[1]]
    training_data = zip(training_inputs, training_results)
    #验证集
    validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
    validation_data = zip(validation_inputs, va_d[1])
    #测试集
    test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
    test_data = zip(test_inputs, te_d[1])
    return (training_data, validation_data, test_data)

def vectorized_result(j):
    #形状为10行1列
    e = np.zeros((10, 1))
    e[j] = 1.0
    return e
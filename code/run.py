# -*- coding: utf-8 -*-
"""
Created on Fri Jun 09 14:36:50 2017

@author: XIE
"""

import mnist_loader
import network

training_data,validation_data, test_data = mnist_loader.load_data_wrapper()

net = network.Network([784, 30, 10])

net.SGD(training_data, 30, 10, 3.0, test_data=test_data)
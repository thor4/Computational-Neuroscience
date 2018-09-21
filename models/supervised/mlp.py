# -*- coding: utf-8 -*-
"""
Created on Mon Sep 10 09:31:24 2018

@author: bryan
"""

import numpy
# sigmoid function from scipy.special
import scipy.special

# define a neural network class that can be initialized, trained and queried
class neuralNetwork:
    # initialize the network 
    def __init__(self, input_nodes, hidden_nodes, output_nodes, learning_rate):
        # set the number of nodes in each layer: input, hidden and output
        self.inodes = input_nodes
        self.hnodes = hidden_nodes
        self.onodes = output_nodes
        # set the learning rate for the network
        self.lrate = learning_rate
        # set the weights for edges connecting each node from each layer
        # weights inside the array are of the form w_ij where origination node 
        # is found in layer i, target node is found in layer j
        # numpy.random.rand selects a number at random [0,1]. subtracting 0.5
        # ensures there are negative numbers to allow for weights to go from 
        # [-0.5,0.5]. convention will be matrices of (j,i). ie: (hidden,input)
        self.w_input_hidden = (numpy.random.rand(self.hnodes,self.inodes) - 0.5)
        self.w_hidden_output = (numpy.random.rand(self.onodes,self.hnodes) - 0.5)
        # define the activation function to use (sigmoid)
        # assign anonymous function lambda as activation_function. have it
        # take in x and return expit(x)
        self.activation_function = lambda x:scipy.special.expit(x)
        pass
    # train the network with a list of inputs and list of targets
    def train(self, inputs_list, targets_list):
        # ensure inputs and targets list is a 2d array (inputs x 1)
        inputs = numpy.array(inputs_list,ndmin=2).T
        targets = numpy.array(targets_list,ndmin=2).T
        # feed-forward inputs through to output layer
        hidden_inputs = numpy.dot(self.w_input_hidden,inputs)
        hidden_outputs = self.activation_function(hidden_inputs)
        final_inputs = numpy.dot(self.w_hidden_output,hidden_outputs)
        final_outputs = self.activation_function(final_inputs)
        # calculate the error (target - prediction)
        output_errors = targets - final_outputs
        # use output errors to determine error at the hidden layer: multiply 
        # each hidden weight by output errors
        hidden_errors = numpy.dot(self.w_hidden_output.T,output_errors)
        # update weights on edges connecting hidden and output layers
        # formula is lrate * error * sig(output) * (1-sig(output)) * hidden
        self.w_hidden_output += self.lrate * numpy.dot((output_errors * final_outputs * (1.0 - final_outputs)), numpy.transpose(hidden_outputs))
        # update weights on edges connecting input and hidden layers
        self.w_input_hidden += self.lrate * numpy.dot((hidden_errors * hidden_outputs * (1.0 - hidden_outputs)), numpy.transpose(inputs))
        pass
    # query the network with list of inputs
    def query(self, inputs_list):
        # ensure inputs list is a 2d array (inputs x 1)
        inputs = numpy.array(inputs_list,ndmin=2).T
        # pass the inputs to the hidden layer by multiplying with each edge's 
        # weight and summing together
        hidden_inputs = numpy.dot(self.w_input_hidden,inputs)
        # add bias?
        # pass hidden inputs through activation function to determine hidden 
        # layer output
        hidden_outputs = self.activation_function(hidden_inputs)
        # pass hidden layer outputs to final output layer by multiplying with 
        # each edge's weight and summing together
        final_inputs = numpy.dot(self.w_hidden_output,hidden_outputs)
        # pass through activation function to determine final output layer
        final_outputs = self.activation_function(final_inputs)
        return final_outputs

# iniitalize neural network object architecture: input,hidden and output layer 
# nodes and learning rate
input_nodes = 2
hidden_nodes = 2
output_nodes = 1
learning_rate = 0.3

# create instance of neural network class
n = neuralNetwork(input_nodes, hidden_nodes, output_nodes, learning_rate)

# query neural network instance with some random inputs
n.query([1.0, 0.5, -1.5])

n.train([1.0, 0.5, -1.5], [2.0, 0.0, 1.0])

n.train([1, 1], [0])
n.train([1, 0], [1])
n.train([0, 1], [1])
n.train([0, 0], [0])

n.query([0, 0])

numpy.random.rand(3,4) - 0.5

inputs_list = [1.0, 0.5, -1.5]
expit()
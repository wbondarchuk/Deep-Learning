import numpy as np

from layers import (
    FullyConnectedLayer, ReLULayer,
    ConvolutionalLayer, MaxPoolingLayer, Flattener,
    softmax_with_cross_entropy, l2_regularization
    )


class ConvNet:
    """
    Implements a very simple conv net

    Input -> Conv[3x3] -> Relu -> Maxpool[4x4] ->
    Conv[3x3] -> Relu -> MaxPool[4x4] ->
    Flatten -> FC -> Softmax
    """
    def __init__(self, input_shape, n_output_classes, conv1_channels, conv2_channels):
        """
        Initializes the neural network

        Arguments:
        input_shape, tuple of 3 ints - image_width, image_height, n_channels
                                         Will be equal to (32, 32, 3)
        n_output_classes, int - number of classes to predict
        conv1_channels, int - number of filters in the 1st conv layer
        conv2_channels, int - number of filters in the 2nd conv layer
        """
        # TODO Create necessary layers
        self.layers = [ConvolutionalLayer(in_channels=input_shape[2], \
                                          out_channels=input_shape[2], \
                                          filter_size=conv1_channels, \
                                          padding=1),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, \
                                       stride=2),
                       ConvolutionalLayer(in_channels=input_shape[2], \
                                          out_channels=input_shape[2], \
                                          filter_size=conv2_channels, \
                                          padding=1),
                       ReLULayer(),
                       MaxPoolingLayer(pool_size=4, \
                                       stride=2),
                       Flattener(),
                       FullyConnectedLayer(n_input=147, \
                                           n_output=n_output_classes)]

    def forward_pass(self, X):
        forward_out = X
        for layer in self.layers:
            forward_out = layer.forward(forward_out)
        return forward_out
    
    def backward_pass(self, d_out):
        
        d_result = d_out
        for layer in reversed(self.layers):
            d_result = layer.backward(d_result)

        return d_result
    
    def compute_loss_and_gradients(self, X, y):
        """
        Computes total loss and updates parameter gradients
        on a batch of training examples
        Arguments:
        X, np array (batch_size, height, width, input_features) - input data
        y, np array of int (batch_size) - classes
        """
        # Before running forward and backward pass through the model,
        # clear parameter gradients aggregated from the previous pass
        for param in self.params().values():
            param.grad = np.zeros_like(param.value)
        # TODO Compute loss and fill param gradients
        # Don't worry about implementing L2 regularization, we will not
        # need it in this assignment
        out = self.forward_pass(X)
        loss, d_out = softmax_with_cross_entropy(out, y)
        self.backward_pass(d_out)
        
        return loss

    def predict(self, X):
        out = self.forward_pass(X)
        pred = np.argmax(out, axis = 1)
        return pred


    def params(self):
        result = {  'W1': self.layers[0].params()['W'],         'B1': self.layers[0].params()['B'], 
                    'W2': self.layers[3].params()['W'],         'B2': self.layers[3].params()['B'], 
                    'W3': self.layers[7].params()['W'],         'B3': self.layers[7].params()['B']}
        return result

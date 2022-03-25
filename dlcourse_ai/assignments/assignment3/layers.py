import numpy as np


def l2_regularization(W, reg_strength):
    '''
    Computes L2 regularization loss on weights and its gradient

    Arguments:
      W, np array - weights
      reg_strength - float value

    Returns:
      loss, single value - l2 regularization loss
      gradient, np.array same shape as W - gradient of weight by l2 loss
    '''
    # TODO: Copy from previous assignment
    loss = reg_strength * np.sum(np.square(W))
    grad = 2 * W * reg_strength 

    return loss, grad


def softmax(predictions):
    '''
    Computes probabilities from scores
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
    Returns:
      probs, np array of the same shape as predictions - 
        probability for every class, 0..1
    '''
    # TODO implement softmax
    # Your final implementation shouldn't have any loops
    
    pred = np.copy(predictions)
    
    if (pred.ndim == 1):
        pred -= np.max(pred)
        pred_exp = np.exp(pred)
        sum_exp = np.sum(pred_exp)
        probs = pred_exp / sum_exp
        
    else:
        pred = (pred.T - np.max(pred, axis = 1)).T
        pred_exp = np.exp(pred)
        sum_exp = np.sum(pred_exp, axis = 1)
        probs = ((pred_exp.T) / sum_exp).T
    return probs


def cross_entropy_loss(probs, target_index):
    '''
    Computes cross-entropy loss
    Arguments:
      probs, np array, shape is either (N) or (batch_size, N) -
        probabilities for every class
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss: single value
    '''
    # TODO implement cross-entropy
    # Your final implementation shouldn't have any loops
    
    if (probs.ndim == 1):
        loss = -np.log(probs[target_index])
    
    else:
        batch_size = probs.shape[0]
        logs = -np.log(probs[range(batch_size), target_index])
        loss = np.sum(logs) / batch_size
    
    return loss


def softmax_with_cross_entropy(preds, target_index):
    """
    Computes softmax and cross-entropy loss for model predictions,
    including the gradient
    Arguments:
      predictions, np array, shape is either (N) or (batch_size, N) -
        classifier output
      target_index: np array of int, shape is (1) or (batch_size) -
        index of the true class for given sample(s)
    Returns:
      loss, single value - cross-entropy loss
      dprediction, np array same shape as predictions - gradient of predictions by loss value
    """
    # TODO: Copy from the previous assignment
    probs = softmax(preds)
    d_preds = probs
    loss = cross_entropy_loss(probs, target_index)
    
    if (probs.ndim == 1):
        d_preds[target_index] -= 1
    else:
        batch_size = preds.shape[0]
        d_preds[range(batch_size), target_index] -= 1
        d_preds /= batch_size

    return loss, d_preds


class Param:
    '''
    Trainable parameter of the model
    Captures both parameter value and the gradient
    '''
    def __init__(self, value):
        self.value = value
        self.grad = np.zeros_like(value)

        
class ReLULayer:
    def __init__(self):
        pass

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.where(X<0,0,X)

    def backward(self, d_out):
        # TODO copy from the previous assignment
        d_result = (self.X > 0) * d_out
        return d_result

    def params(self):
        return {}


class FullyConnectedLayer:
    def __init__(self, n_input, n_output):
        self.W = Param(0.001 * np.random.randn(n_input, n_output))
        self.B = Param(0.001 * np.random.randn(1, n_output))
        self.X = None

    def forward(self, X):
        # TODO copy from the previous assignment
        self.X = X
        return np.dot(X, self.W.value) + self.B.value

    def backward(self, d_out):
        # TODO copy from the previous assignment
        
        self.W.grad = np.dot(self.X.T, d_out)
        self.B.grad = np.sum(d_out, axis=0).reshape(1, -1)
        d_result = np.dot(d_out, self.W.value.T)

        return d_result       

    def params(self):
        return { 'W': self.W, 'B': self.B }



    
class ConvolutionalLayer:
    def __init__(self, in_channels, out_channels,
                 filter_size, padding):
        '''
        Initializes the layer
        
        Arguments:
        in_channels, int - number of input channels
        out_channels, int - number of output channels
        filter_size, int - size of the conv filter
        padding, int - number of 'pixels' to pad on each side
        '''

        self.filter_size = filter_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.W = Param(
            np.random.randn(filter_size, filter_size,
                            in_channels, out_channels)
        )

        self.B = Param(np.zeros(out_channels))
       
        self.padding = padding
        self.X = None


    def forward(self, X):
        batch_size, height, width, channels = X.shape

        out_height = 0
        out_width = 0
        
        X_with_pad = np.zeros((batch_size , height + 2 * self.padding , width + 2 * self.padding , channels))      
        X_with_pad[: , self.padding: X_with_pad.shape[1]-self.padding , self.padding:X_with_pad.shape[2]-self.padding , :] = X
        self.X = X_with_pad 
        
        out_height = X_with_pad.shape[1] - self.filter_size + 1
        out_width = X_with_pad.shape[2] - self.filter_size + 1       
        output = np.zeros((batch_size , out_height , out_width , self.out_channels))     
        
        # TODO: Implement forward pass
        # Hint: setup variables that hold the result
        # and one x/y location at a time in the loop below
        
        # It's ok to use loops for going over width and height
        # but try to avoid having any other loops
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement forward pass for specific location
                X_matrix = X_with_pad[:, y: y + self.filter_size, x:x + self.filter_size, :]                
                X_matrix_arr = X_matrix.reshape(batch_size, self.filter_size*self.filter_size * self.in_channels)            
                W_arr = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
                Res_arr = np.dot(X_matrix_arr , W_arr) + self.B.value          
                Res_mat = Res_arr.reshape(batch_size, 1, 1, self.out_channels)              
                output[: , y: y + self.filter_size , x:x + self.filter_size, :] = Res_mat 
        
        return output


    def backward(self, d_out):
        # Hint: Forward pass was reduced to matrix multiply
        # You already know how to backprop through that
        # when you implemented FullyConnectedLayer
        # Just do it the same number of times and accumulate gradients

        batch_size, height, width, channels = self.X.shape
        _, out_height, out_width, out_channels = d_out.shape
        
        dX = np.zeros((batch_size, height, width, channels))
        W_arr = self.W.value.reshape(self.filter_size * self.filter_size * self.in_channels, self.out_channels)
        
        # TODO: Implement backward pass
        # Same as forward, setup variables of the right shape that
        # aggregate input gradient and fill them for every location
        # of the output

        # Try to avoid having any other loops here too
        for y in range(out_height):
            for x in range(out_width):
                # TODO: Implement backward pass for specific location
                # Aggregate gradients for both the input and
                # the parameters (W and B)
                
                X_local_mat = self.X[:, x:x + self.filter_size , y:y + self.filter_size, :]           
                X_arr = X_local_mat.reshape(batch_size, self.filter_size * self.filter_size * self.in_channels)
                d_local = d_out[:, x:x + 1, y:y + 1, :]
                dX_arr = np.dot(d_local.reshape(batch_size, -1), W_arr.T)
                dX[:, x:x + self.filter_size , y:y + self.filter_size, :] += dX_arr.reshape(X_local_mat.shape)
                dW = np.dot(X_arr.T, d_local.reshape(batch_size, -1))
                dB = np.dot(np.ones((1, d_local.shape[0])), d_local.reshape(batch_size, -1))
                self.W.grad += dW.reshape(self.W.value.shape)
                self.B.grad += dB.reshape(self.B.value.shape)
        
        return dX[:, self.padding : (height - self.padding), self.padding : (width - self.padding), :]


    def params(self):
        return { 'W': self.W, 'B': self.B }


class MaxPoolingLayer:
    def __init__(self, pool_size, stride):
        '''
        Initializes the max pool

        Arguments:
        pool_size, int - area to pool
        stride, int - step size between pooling windows
        '''
        self.pool_size = pool_size
        self.stride = stride
        self.X = None
        self.masks = {}

    def forward(self, X):
        batch_size, height, width, channels = X.shape
        self.X = X
        self.masks.clear()
        
        # TODO: Implement maxpool forward pass
        # Hint: Similarly to Conv layer, loop on
        # output x/y dimension
        
        out_height = int((height - self.pool_size) / self.stride + 1)
        out_width = int((width - self.pool_size) / self.stride + 1)
        
        output = np.zeros((batch_size, out_height, out_width, channels))
        
        mult = self.stride
        
        for x in range(out_width):
            for y in range(out_height):
                I = X[:, x*mult:x*mult+self.pool_size, y*mult:y*mult+self.pool_size, :]
                self.mask(x=I, pos=(x, y))
                output[:, x, y, :] = np.max(I, axis=(1, 2))
        return output

    def backward(self, d_out):
        # TODO: Implement maxpool backward pass
        _, out_height, out_width, _ = d_out.shape
        dX = np.zeros_like(self.X)
        mult = self.stride
        
        for x in range(out_width):
            for y in range(out_height):
                dX[:, x * mult:x * mult + self.pool_size, y * mult:y * mult + self.pool_size, :] += d_out[:, x:x + 1, y:y + 1, :] * self.masks[(x, y)]  
        return dX

    def mask(self, x, pos):
        zero_mask = np.zeros_like(x)
        batch_size, height, width, channels = x.shape
        x = x.reshape(batch_size, height * width, channels)
        idx = np.argmax(x, axis=1)

        n_idx, c_idx = np.indices((batch_size, channels))
        zero_mask.reshape(batch_size, height * width, channels)[n_idx, idx, c_idx] = 1
        self.masks[pos] = zero_mask    
    
    
    def params(self):
        return {}


class Flattener:
    def __init__(self):
        self.X_shape = None

    def forward(self, X):
        batch_size, height, width, channels = X.shape

        # TODO: Implement forward pass
        # Layer should return array with dimensions
        # [batch_size, hight*width*channels]
        self.X_shape = X.shape
        return X.reshape(batch_size, height * width * channels)

    def backward(self, d_out):
        # TODO: Implement backward pass
        return d_out.reshape(self.X_shape)

    def params(self):
        # No params!
        return {}

### Nick Newman Homework 2

## Problem 1: Coding a single layer neural network without using a Machine Learning
## or Deep Learning framework

#%%
import os 
import struct
import numpy as np
import gzip
import matplotlib.pyplot as plt
from textwrap import wrap

#%% This is only to open unzip the gzip files
#for filename in ['t10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz', 'train-images-idx3-ubyte.gz','train-labels-idx1-ubyte.gz']:
#    with gzip.open(filename, 'rb') as f_in:
#        with open(filename[:-3], 'wb') as f_out:
#            shutil.copyfileobj(f_in, f_out)

#%%
            
def read_idx(filename):
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)
    
X_train = read_idx('train-images-idx3-ubyte')
y_train = read_idx('train-labels-idx1-ubyte')
X_test = read_idx('t10k-images-idx3-ubyte')
y_test = read_idx('t10k-labels-idx1-ubyte')


#%%

# Showing graphs with samples of the handwritten numbers
for i in np.unique(y_train):
    plt.imshow(X_train[y_train==i][0], cmap="binary")
    print(i)
    plt.show()

#%% 

class MLP:
    """ Defining a One-Hidden Layer Neural Network Model (Multi-Layer Perceptron)
        for classification wit Mini-Batch Gradient Descent
    
    The parameters are:
        n_iter: number of iterations over the dataset
        eta: learning rate/step value
        n_hidden: number of hidden neurons in the network
        hidden_act: activation function that is used on the hidden layer
        encode: if the target values are not in binary form, then this 
        will one hot encode them
        batch_size: number of samples used in each batch
        
    """
    
    def __init__(self, n_iter=1000, eta=0.05, n_hidden=10, hidden_act = 'sigmoid',
                 encode=True, batch_size=1):
        self.n_iter = n_iter
        self.eta = eta
        self.encode=encode
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.hidden_act = hidden_act
            
    def hidden_layer(self, z):
        # defining which activation function gets used in the hidden layer
        if self.hidden_act=='relu':
            return np.maximimum(z, 0)
        else:
            return 1/(1+ np.exp(-z))

    def softmax(self, z):
        # This is the softmax function of z_out
        # used to get the predicted class for a multi-class problem
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape(-1,1)
    
    def onehot(self, data):
        """ This takes in target values and returns them coded in binary for each
            distinct class. The index values don't necessarily correspond to the
            original target values as the values are mapped into a distinct range.
        """
        n_classes = np.unique(data)
        onehot_array = np.zeros((data.shape[0], np.max(data)+1))
        onehot_array[np.arange(data.shape[0]), data] = 1
        if n_classes.shape[0] > 2:
            return onehot_array[:,n_classes]     
        # return only one column if there are only two target variable to avoid
        # redundancy. This reflects the first instance (ordinal) in the data.
        else:
            return onehot_array[:,n_classes[0]]
        
    
    def forward_prop(self, X):
        # forward propogation through the neural network
        
        # [num in X, num of features] dot [num of features, n hidden units] + [hidden bias]
        # -> [num in X, n hidden units] 
        Z_h = np.dot(X, self.Wh) + self.bias_h
        
        # Hidden layer activation function
        # [num in X, n hidden units] 
        A_h = self.hidden_layer(Z_h)
        
        # Z output of hidden layer
        # [num in X,n hidden units] dot [num hidden units, n classes]
        # -> [num in X, n classes]
        Z_out = np.dot(A_h, self.Wout) + self.bias_out
        
        # output layer activation function
        # [num in X, n classes]
        A_out = self.softmax(Z_out)
        return Z_h, A_h, Z_out, A_out


    def backprop(self, Z_h, A_h, Z_out, A_out, X_train, y_train):    
        # backpropagating through the network to determine the gradients
        
        # [num in X, n classes]
        output_error = A_out - y_train
        
        # [n hidden units, num in X] dot [num in X, n classes]
        # -> [n hidden units, n classes]
        DJ_DWout = np.dot(A_h.T, output_error)
        
        # [num in X, n hidden units] 
        sigmoid_deriv = A_h * (1-A_h)
        
        # [num in X, n classes] dot [n classes, num hidden units]
        # * [num in X, n hidden units] -> [num in X, n hidden units]
        DJ_DWhid = np.dot(output_error, self.Wout.T) * sigmoid_deriv
        
        # gradient of layer 1 W and layer 1 bias
        # [n hidden units, num in X] dot [num in X, n classes]
        # -> [n hidden units, n classes]
        grad_Whid = np.dot(X_train.T, DJ_DWhid)
        grad_Bhid = np.sum(DJ_DWhid, axis=0)
        
        # gradient of layer 2 W and layer 2 bias
        # [num of features, num in X] dot [num in X, n hidden units]
        # -> [num of features, n hidden units]
        grad_Wout = DJ_DWout
        grad_Bout = np.sum(output_error, axis=0)
        
        # updating the gradients
        self.Wh -= grad_Whid * self.eta
        self.bias_h -= grad_Bhid * self.eta
        
        self.Wout -= grad_Wout * self.eta
        self.bias_out -= grad_Bout * self.eta
        
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        # fit the training data
        if self.encode:
            y_train_oh = self.onehot(y_train)
        n_classes = y_train_oh.shape[1]
        n_features = X_train.shape[1]
        
        # initialize the values of the weights to random normal values
        self.Wh = np.random.randn(n_features, self.n_hidden)
        self.Wout = np.random.randn(self.n_hidden, n_classes)
        
        # initialize the values of the biases to zero
        self.bias_h = np.zeros((1,self.n_hidden))
        self.bias_out = np.zeros((1,n_classes))
        
        # m = number of samples
        m = y_train.shape[0]

        # dictionary to keep track of the values as the network goes through
        # the iterations
        self.metrics = {'cost': [],'train_accuracy': [], 'valid_accuracy':[]}

        for i in range(self.n_iter):
            
            shuffled_values = np.random.permutation(m)
            X_shuffled = X_train[shuffled_values]
            y_shuffled = y_train_oh[shuffled_values]
            for batch in range(0, m, self.batch_size):
                x_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]
                
                # forward propagation
                Z_h, A_h, Z_out, A_out = self.forward_prop(x_batch)
                
                # backpropagation
                self.backprop(Z_h, A_h, Z_out, A_out, x_batch, y_batch)
            
            # After each iteration, do an evaluation
            # Evaluating on the training set
            Z_h, A_h, Z_out, A_out = self.forward_prop(X_train)
            cost = self.cost_function(A_out, y_train_oh)
            train_predictions = self.predict(A_out)
            
            # Evaluating on the validation set
            Z_h, A_h, Z_out, A_out = self.forward_prop(X_valid)
            valid_predictions = self.predict(A_out)

            train_accuracy = np.sum(train_predictions == y_train).astype(np.float)/train_predictions.shape[0]
            valid_accuracy = np.sum(valid_predictions == y_valid).astype(np.float)/valid_predictions.shape[0]
            
            if not (i+1)%10:
                print("Iteration: {}\t Train Acc: {:.3f}\t Validation Acc: {:.3f}".format(i+1, train_accuracy, valid_accuracy))
            
            self.metrics['cost'].append(cost)
            self.metrics['train_accuracy'].append(train_accuracy)
            self.metrics['valid_accuracy'].append(valid_accuracy)
            
        return self
            
    def cost_function(self, A_out, y):
       return np.average(-y*np.log(A_out) - ((1-y)*np.log(1-A_out)))
        
   
    def predict(self, output):
        # return the value with the highest percentage
        return np.argmax(output, axis=1)
        

#%% 
## Problem 2: Coding a single layer neural network without using a Machine Learning
## or Deep Learning framework and using L2 Regularization


class MLP_l2:
    """ Defining a One-Hidden Layer Neural Network Model (Multi-Layer Perceptron)
        for classification with l2 regularization and Mini-Batch Gradient Descent
    
    The parameters are: 
        l2: l2 regularization rate
        n_iter: number of iterations over the dataset
        eta: learning rate/step value
        n_hidden: number of hidden neurons in the network
        hidden_act: activation function that is used on the hidden layer
        encode: if the target values are not in binary form, then this 
        will one hot encode them
        batch_size: number of samples used in each batch
    """

    
    def __init__(self, l2=0.0, n_iter=1000, eta=0.05, n_hidden=10, hidden_act = 'sigmoid',
                 encode=True, batch_size=1):
        self.l2 = l2
        self.n_iter = n_iter
        self.eta = eta
        self.encode=encode
        self.n_hidden = n_hidden
        self.batch_size = batch_size
        self.hidden_act = hidden_act
            
    def hidden_layer(self, z):
        if self.hidden_act=='relu':
            return np.maximimum(z, 0)
        else:
            return 1/(1+ np.exp(-z))

    
    def softmax(self, z):
        # This is the softmax function of z_out
        # used to get the predicted class for a multi-class problem
        return np.exp(z)/np.sum(np.exp(z), axis=1).reshape(-1,1)
    
    def onehot(self, data):
        """ This takes in target values and returns them coded in binary for each
            distinct class. The index values don't necessarily correspond to the
            original target values as the values are mapped into a distinct range.
        """
        n_classes = np.unique(data)
        onehot_array = np.zeros((data.shape[0], np.max(data)+1))
        onehot_array[np.arange(data.shape[0]), data] = 1
        if n_classes.shape[0] > 2:
            return onehot_array[:,n_classes]     
        # return only one column if there are only two target variable to avoid
        # redundancy. This reflects the first instance (ordinal) in the data.
        else:
            return onehot_array[:,n_classes[0]]
        
    
    def forward_prop(self, X):
        # forward propogation through the neural network
        
        # [num in X, num of features] dot [num of features, n hidden units] + [hidden bias]
        # -> [num in X, n hidden units] 
        Z_h = np.dot(X, self.Wh) + self.bias_h
        
        # Hidden layer activation function
        # [num in X, n hidden units] 
        A_h = self.hidden_layer(Z_h)
        
        # Z output of hidden layer
        # [num in X,n hidden units] dot [num hidden units, n classes]
        # -> [num in X, n classes]
        Z_out = np.dot(A_h, self.Wout) + self.bias_out
        
        # output layer activation function
        # [num in X, n classes]
        A_out = self.softmax(Z_out)
        return Z_h, A_h, Z_out, A_out


    def backprop(self, Z_h, A_h, Z_out, A_out, X_train, y_train):       
        # [num in X, n classes]
        output_error = A_out - y_train
        
        # [n hidden units, num in X] dot [num in X, n classes]
        # -> [n hidden units, n classes]
        DJ_DWout = np.dot(A_h.T, output_error)
        
        # [num in X, n hidden units] 
        sigmoid_deriv = A_h * (1-A_h)
        
        # [num in X, n classes] dot [n classes, num hidden units]
        # * [num in X, n hidden units] -> [num in X, n hidden units]
        DJ_DWhid = np.dot(output_error, self.Wout.T) * sigmoid_deriv
        
        # gradient of layer 1 W and layer 1 bias
        # [n hidden units, num in X] dot [num in X, n classes]
        # -> [n hidden units, n classes]
        grad_Whid = np.dot(X_train.T, DJ_DWhid)
        grad_Bhid = np.sum(DJ_DWhid, axis=0)
        
        # gradient of layer 2 W and layer 2 bias
        # [num of features, num in X] dot [num in X, n hidden units]
        # -> [num of features, n hidden units]
        grad_Wout = DJ_DWout
        grad_Bout = np.sum(output_error, axis=0)
        
        
        ## Regularization \
        # we don't add regularization to the bias terms
        l2_Whid = self.Wh * self.l2        
        l2_Wout = self.Wout * self.l2
        
        self.Wh -= (grad_Whid + l2_Whid) * self.eta
        self.bias_h -= grad_Bhid * self.eta
        
        self.Wout -= (grad_Wout * self.eta) + l2_Wout
        self.bias_out -= grad_Bout * self.eta
    
        
        
    def fit(self, X_train, y_train, X_valid, y_valid):
        # fit the training data
        if self.encode:
            y_train_oh = self.onehot(y_train)
        n_classes = y_train_oh.shape[1]
        n_features = X_train.shape[1]
        
        # initialize the values of the weights to random normal values
        self.Wh = np.random.randn(n_features, self.n_hidden)
        self.Wout = np.random.randn(self.n_hidden, n_classes)
        
        # initialize the values of the biases to zero
        self.bias_h = np.zeros((1,self.n_hidden))
        self.bias_out = np.zeros((1,n_classes))
        
        # m = number of samples
        m = y_train.shape[0]

        # dictionary to keep track of the values as the network goes through
        # the iterations
        self.metrics = {'cost': [],'train_accuracy': [], 'valid_accuracy':[]}

        for i in range(self.n_iter):
            
            shuffled_values = np.random.permutation(m)
            X_shuffled = X_train[shuffled_values]
            y_shuffled = y_train_oh[shuffled_values]
            for batch in range(0, m, self.batch_size):
                x_batch = X_shuffled[batch:batch+self.batch_size]
                y_batch = y_shuffled[batch:batch+self.batch_size]
                
                # forward propagation
                Z_h, A_h, Z_out, A_out = self.forward_prop(x_batch)
                
                # backpropagation
                self.backprop(Z_h, A_h, Z_out, A_out, x_batch, y_batch)
            
            # After each iteration, do an evaluation
            # Evaluating on the training set
            Z_h, A_h, Z_out, A_out = self.forward_prop(X_train)
            cost = self.cost_function(A_out, y_train_oh)
            train_predictions = self.predict(A_out)
            
            # Evaluating on the validation set
            Z_h, A_h, Z_out, A_out = self.forward_prop(X_valid)
            valid_predictions = self.predict(A_out)

            train_accuracy = np.sum(train_predictions == y_train).astype(np.float)/train_predictions.shape[0]
            valid_accuracy = np.sum(valid_predictions == y_valid).astype(np.float)/valid_predictions.shape[0]
            
            if not (i+1)%10:
                print("Iteration: {}\t Train Acc: {:.3f}\t Validation Acc: {:.3f}".format(i+1, train_accuracy, valid_accuracy))
            
            self.metrics['cost'].append(cost)
            self.metrics['train_accuracy'].append(train_accuracy)
            self.metrics['valid_accuracy'].append(valid_accuracy)
            
        return self
            
    def cost_function(self, A_out, y):
        m = y.shape[0]
        l2_cost = (self.l2/(2*m))*(np.sum(self.Wh**2)+np.sum(self.Wout**2))
        return np.average(-y*np.log(A_out) - ((1-y)*np.log(1-A_out))) + l2_cost
   
    def predict(self, output):
        # return the value with the highest percentage
        return np.argmax(output, axis=1)

#%%
        
## Scaling the values so that the model can more accurately adjust the weights
X_train = X_train.astype(np.float)/255.
X_test = X_test.astype(np.float)/255.

## Reshaping the arrays so that they can be input accordingly
X_train = X_train.reshape(-1,784)
X_test = X_test.reshape(-1,784)


# Running a NN without L2 Regularization
model = MLP(n_iter=300,n_hidden=50, batch_size=500, eta=0.001)

model.fit(X_train=X_train[:50000], 
       y_train=y_train[:50000],
       X_valid=X_train[50000:],
       y_valid=y_train[50000:])


# Running a NN with L2 Regularization
model = MLP_l2(n_iter=300,n_hidden=50, batch_size=500, eta=0.001, l2=0.1)

model.fit(X_train=X_train[:50000], 
       y_train=y_train[:50000],
       X_valid=X_train[50000:],
       y_valid=y_train[50000:])  
   




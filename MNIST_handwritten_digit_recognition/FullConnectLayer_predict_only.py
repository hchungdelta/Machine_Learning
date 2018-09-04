from __future__ import print_function
import json
import numpy as np 
import tensorflow as tf
import os
# A safe call, in order not to build the graph with the system's default# 
tf.reset_default_graph() 

 
def load(filename):
    # use json to save and load the handwritten data from hand-written penal)
    print("Loading", filename ,"...")
    f = open(filename, "r")
    data = json.load(f)
    f.close()    
    x_input=np.array(data["x_input"])
    return x_input 



# Parameters
#dropout= 1

# Restore in your current directory
name_your_save_file="model"
model_path = os.getcwd()+str("/")+str(name_your_save_file) 

# input your handwritten digit data (input only).
x_HW = load('test.json')  # I set default name as test.json    
x_HW=x_HW.reshape(-1,784)

# Network Parameters
n_input = 784     # MNIST data input (img shape: 28*28)
n_hidden_1 = 512  # Amount of neurals in 1st (hidden) layer  
n_hidden_2 = 1024 # Amount of neurals in 2nd (hidden) layer  
n_classes = 10    # MNIST classification (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_classes])

# Create model
def multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([n_input, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
pred = multilayer_perceptron(x, weights, biases)
 

# Initialize the variables (i.e. assign their default value)

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

print("Starting predict session...")
with tf.Session() as sess:
    
    # Restore model weights from previously saved model
    saver.restore(sess, model_path)
    save_path = saver.save(sess, model_path)   
    prediction=tf.argmax(pred,1)
    result = prediction.eval(feed_dict={x: x_HW}) 
    print("=================My hand written programme======================")
    print("Prediction :" ,result)
    print("=================My hand written programme======================")

def Get_result():
    return result
        
    
    
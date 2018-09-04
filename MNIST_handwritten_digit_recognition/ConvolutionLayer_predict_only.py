from __future__ import print_function
import json
import numpy as np 
import tensorflow as tf
import os 

tf.reset_default_graph()


def load(filename):
    print("Loading", filename ,"...")
    f = open(filename, "r")
    data = json.load(f)
    f.close()    
    x_input=np.array(data["x_input"])
    return x_input 


# Switch on/off the train_process and predict_only

# Parameters


# Restore in your current directory
name_your_save_file="modelCNN"
model_path = os.getcwd()+str("/")+str(name_your_save_file) 


# Network Parameters (CNN)
CNN_grid_size=4
CNN_input_dimension=1   #   due to monochrome. If input is RGB, this parameter needed to be set as 3. 
CNN_filter_amount_1=8
CNN_filter_amount_2=4
# Network Parameters (FNN)
n_hidden_0 = 49   # CNN (28x28)-> "2x2" max_pool ->  "2x2" max_pool -> FNN (7x7)
n_hidden_1 = 98   # Amount of neurals in 1st (hidden) layer  
n_hidden_2 = 196  # Amount of neurals in 2nd (hidden) layer  

n_input = 784 # MNIST data input (img shape: 28*28)
n_classes = 10 # MNIST total classes (0-9 digits)

# tf Graph input
x = tf.placeholder("float", [None, n_input ])
y = tf.placeholder("float", [None, n_classes])
x_HW = load('test.json')
x_HW=x_HW.reshape(-1,784)

 

#mnist.train.images=mnist.train.images.reshape(-1,28,28,1)
 
# Create model
def multilayer_perceptron(x, weights, biases):
    x = tf.reshape(x, shape=[-1, 28, 28, 1])
    # Convolution layer with RELU activation / mal_pool
    
    'CNN : 28x28 --> 14x14  '
    cn_layer_1 = tf.nn.conv2d(x,  weights['cn1'], strides=[1, 1, 1, 1], padding='SAME' )
    cn_layer_1 = tf.nn.bias_add(cn_layer_1, biases['cn1'])
    cn_layer_1 = tf.nn.relu(cn_layer_1)
    cn_layer_1 = tf.nn.max_pool(cn_layer_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
    'CNN : 14x14 --> 7x7  '  
    cn_layer_2 = tf.nn.conv2d(cn_layer_1,  weights['cn2'], strides=[1, 1, 1, 1], padding='SAME' )
    cn_layer_2 = tf.nn.bias_add(cn_layer_2, biases['cn2'])
    cn_layer_2 = tf.nn.relu(cn_layer_2)
    cn_layer_2 = tf.nn.max_pool(cn_layer_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1],padding='SAME')
       
    cn_layer_2 = tf.layers.flatten(cn_layer_2)
    # Hidden layer with RELU activation    
    layer_1 = tf.add(tf.matmul(cn_layer_2, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)

    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)

    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer

# Store layers weight & bias
weights = {

    'cn1': tf.Variable(tf.random_normal([CNN_grid_size, CNN_grid_size, CNN_input_dimension, CNN_filter_amount_1])),
    'cn2': tf.Variable(tf.random_normal([CNN_grid_size, CNN_grid_size, CNN_filter_amount_1, CNN_filter_amount_2])),
    'h1': tf.Variable(tf.random_normal([n_hidden_0*CNN_filter_amount_2, n_hidden_1])),
    'h2': tf.Variable(tf.random_normal([n_hidden_1, n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_hidden_2, n_classes]))
}
biases = {
    'cn1': tf.Variable(tf.random_normal([CNN_filter_amount_1])),
    'cn2': tf.Variable(tf.random_normal([CNN_filter_amount_2])),
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'b2': tf.Variable(tf.random_normal([n_hidden_2])),
    'out': tf.Variable(tf.random_normal([n_classes]))
}

# Construct model
keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
pred = multilayer_perceptron(x, weights, biases)
output_softmax = tf.nn.softmax(pred)



# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

print("Starting predict your_test session...")
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
                    
        
    
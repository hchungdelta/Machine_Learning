'''
9/4/2018 
Author : Hao-Chien, Hung

A demo for handwritten digit recognition.
-Construct a neural network with Tensorflow
-handwritten digit data are from : mnist 

This code package including 3 python files:
1. FullConnectLayer 
2. ConvolutionLayer
3. Writing Panel  (#need to import pygame)

Writing Panel enables you to write down the number on the penal 
(just a simple version of MSprint in python)
And use 1. or 2. to predict your handwritten digit directly.

Already trained data are available in this package (while not so accurate).
You can use it directly.
It is highly recommends to train the network yourself.

The basic neural network framework is based on the work of Aymeric Damien  
(https://github.com/aymericdamien/TensorFlow-Examples/)
'''
from __future__ import print_function
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
import json
import numpy as np 
import tensorflow as tf
import os 


# (optional) tqdm is a tool to visualize the current progress.
# it can print out the text like:     |######          | 35%
try:
    from tqdm import tqdm
    improt_tqdm = True
except ImportError:
    improt_tqdm = False
    pass

tf.reset_default_graph()




def load(filename):
    print("Loading", filename ,"...")
    f = open(filename, "r")
    data = json.load(f)
    f.close()    
    x_input=np.array(data["x_input"])
    return x_input 


# Switch on/off the train_process and predict_only
train_session = True
predict_minst_test_session = False
predict_your_test_session = False

# Parameters
learning_rate = 0.001
batch_size = 100
display_step = 1
dropout= 1

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

test_X=  mnist.test.images.reshape(-1,28,28,1)
train_X= mnist.train.images.reshape(-1,28,28,1)
 

#mnist.train.images=mnist.train.images.reshape(-1,28,28,1)
 
# Create model
def multilayer_perceptron(x, weights, biases, dropout):
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
    layer_1 = tf.nn.dropout(layer_1, dropout)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout)
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
pred = multilayer_perceptron(x, weights, biases, keep_prob)
output_softmax = tf.nn.softmax(pred)
# Define loss and optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=pred, labels=y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# 'Saver' op to save and restore all the variables
saver = tf.train.Saver()

 
if train_session  :
    print("Starting train session...")
    with tf.Session() as sess:
 
        # Run the initializer
        sess.run(init)
    
        # Training cycle       
        for epoch in range(20):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            if improt_tqdm == True :  pbar = tqdm(total=total_batch+1)
            # Loop over all batches
            for i in range(total_batch):
                if i%50 ==0 and improt_tqdm == True : pbar.update(50)
                if i%50 ==0 and improt_tqdm == False : print("completed : " ,i,"/", total_batch)
                batch_x, batch_y = mnist.train.next_batch(batch_size) 
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y, keep_prob : 1.0})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if improt_tqdm == True :  pbar.close()
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
 
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels , keep_prob : 1.0}))
        
        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

 
if predict_minst_test_session  :
    print("Starting predict minst_test session...")
    with tf.Session() as sess:
        sess.run(init) # initialization 
    
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        save_path = saver.save(sess, model_path)
        print("Model restored from file: %s" % save_path) 

        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
        x_test=mnist.test.images[:10] ; y_test=mnist.test.labels[:10]
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))

    
    
        prediction=tf.argmax(pred,1)
        print("=============================================================")
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels , keep_prob : 1.0}))
        print("The first 10 digits in the minit.test : ")
        print("Prediction :" ,prediction.eval(feed_dict={x: x_test , keep_prob : 1.0})        )
        print("Labels     :"  ,sess.run(tf.argmax(y_test, 1))               )
        print("=============================================================")
    

if predict_your_test_session  :
    print("Starting predict your_test session...")
    with tf.Session() as sess:
        sess.run(init) # initialization 
    
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        save_path = saver.save(sess, model_path)
        print("Model restored from file: %s" % save_path) 
       
        prediction=tf.argmax(pred,1)
    
        print("=================My hand written programme======================")
        print("Prediction :" ,prediction.eval(feed_dict={x: x_HW, keep_prob : 1.0})        )
        print("=================My hand written programme======================")
        
        
            
        
    
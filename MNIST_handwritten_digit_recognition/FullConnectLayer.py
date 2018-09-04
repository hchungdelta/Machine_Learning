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

# Switch on/off the train_process and predict_only
train_session = False
predict_minst_test_session = False
predict_your_test_session = True  

# Parameters
learning_rate = 0.001
batch_size   = 100
display_step = 1
dropout= 1

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
    layer_1 = tf.nn.dropout(layer_1, dropout)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, dropout)
    # Output layer with linear activation
    out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
    return out_layer
def test_multilayer_perceptron(x, weights, biases):
    # Hidden layer with RELU activation
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    layer_1 = tf.nn.relu(layer_1)
    layer_1 = tf.nn.dropout(layer_1, 1)
    # Hidden layer with RELU activation
    layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
    layer_2 = tf.nn.relu(layer_2)
    layer_2 = tf.nn.dropout(layer_2, 1)
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
pred = multilayer_perceptron(x, weights, biases )
test_pred = test_multilayer_perceptron(x, weights, biases )
output_softmax = tf.nn.softmax(test_pred)
 
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
        sess.run(init)  # initialization 
        # Training cycle
        for epoch in range(40):
            avg_cost = 0.
            total_batch = int(mnist.train.num_examples/batch_size)
            # Loop over all batches
            for i in range(total_batch):
                batch_x, batch_y = mnist.train.next_batch(batch_size)
                # Run optimization op (backprop) and cost op (to get loss value)
                _, c = sess.run([optimizer, cost], feed_dict={x: batch_x,y: batch_y})

                # Compute average loss
                avg_cost += c / total_batch
            # Display logs per epoch step
            if epoch % display_step == 0:
                print("Epoch:", '%04d' % (epoch+1), "cost=", \
                    "{:.9f}".format(avg_cost))
        print("Optimization Finished!")
    
        # Test model
        correct_prediction = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
 
        # Calculate accuracy
        accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        
        # Save model weights to disk
        save_path = saver.save(sess, model_path)
        print("Model saved in file: %s" % save_path)

 
if predict_minst_test_session  :
    print("Starting predict session...")
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
        print("Accuracy:", accuracy.eval({x: mnist.test.images, y: mnist.test.labels}))
        print("The first 10 digits in the minit.test : ")
        print("Prediction :" ,prediction.eval(feed_dict={x: x_test})        )
        print("Labels     :"  ,sess.run(tf.argmax(y_test, 1))               )
        print("=============================================================")
    

if predict_your_test_session  :
    print("Starting predict session...")
    with tf.Session() as sess:
        sess.run(init) # initialization 
    
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        save_path = saver.save(sess, model_path)
        print("Model restored from file: %s" % save_path) 
       
        prediction=tf.argmax(pred,1)
    
        print("=================My hand written programme======================")
        print("Prediction :" ,prediction.eval(feed_dict={x: x_HW}       )
        print("=================My hand written programme======================")

            
        
    
    
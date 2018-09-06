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
import json
import numpy as np 
import tensorflow as tf
import os
# A safe call, in order not to build the graph with the system's default# 
#tf.reset_default_graph() 

 
def load(filename):
    # use json to save and load the handwritten data from hand-written penal)
    print("Loading", filename ,"...")
    f = open(filename, "r")
    data = json.load(f)
    f.close()    
    x_input=np.array(data["x_input"])
    return x_input 




class FullConnectLayer() :
    def __init__(self) : 
        # Parameters
        tf.reset_default_graph() 
        self.learning_rate = 0.001
        self.batch_size   = 100
        self.display_step = 1
        self.dropout= 1
        self.epoch = 20 

        # create a subdirectory to save/restore the data  
        self.name_your_save_file="model"
        self.model_path = os.getcwd()+'\\FCL_data\\'+str(self.name_your_save_file) 

        # Network Parameters
        self.n_input = 784     # MNIST data input (img shape: 28*28)
        self.n_hidden_1 = 512  # Amount of neurals in 1st (hidden) layer  
        self.n_hidden_2 = 1024 # Amount of neurals in 2nd (hidden) layer  
        self.n_classes = 10    # MNIST classification (0-9 digits)

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
            'h1': tf.Variable(tf.random_normal([self.n_input, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }
        self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases )
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)        
        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()
        self.saver = tf.train.Saver()

    # Create model
    def multilayer_perceptron(self, x, weights, biases):
        # Hidden layer with RELU activation
        layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
        layer_1 = tf.nn.relu(layer_1)
        layer_1 = tf.nn.dropout(layer_1, self.dropout)
        # Hidden layer with RELU activation
        layer_2 = tf.add(tf.matmul(layer_1, weights['h2']), biases['b2'])
        layer_2 = tf.nn.relu(layer_2)
        layer_2 = tf.nn.dropout(layer_2, self.dropout)
        # Output layer with linear activation
        out_layer = tf.matmul(layer_2, weights['out']) + biases['out']
        return out_layer


    def train(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        print("Starting train session...")
        with tf.Session() as sess:   
            sess.run(self.init)  # initialization 
            # Training cycle
            for epoch in range(self.epoch):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/self.batch_size)
                # Loop over all batches
                for i in range(total_batch):
                    batch_x, batch_y = mnist.train.next_batch(self.batch_size)
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x,self.y: batch_y})
    
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
        
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
     
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels}))
            
            # Save model weights to disk
            save_path = self.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path)
    def predict_mnist(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        with tf.Session() as sess:
            sess.run(self.init) # initialization 
        
            # Restore model weights from previously saved model
            self.saver.restore(sess, self.model_path)
            save_path = self.saver.save(sess, self.model_path)
            print("Model restored from file: %s" % save_path) 
    
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
            x_test=mnist.test.images[:10] ; y_test=mnist.test.labels[:10]
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
    
        
        
            prediction=tf.argmax(self.pred,1)
            print("=============================================================")
            print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels}))
            print("The first 10 digits in the minit.test : ")
            print("Prediction :" ,prediction.eval(feed_dict={self.x: x_test})        )
            print("Labels     :"  ,sess.run(tf.argmax(y_test, 1))               )
            print("=============================================================")
    
    def predict_test(self):
        # input your handwritten digit data from writing panel (input only).
        x_HW = load('test.json')  # I set default name as test.json    
        x_HW=x_HW.reshape(-1,784)

        with tf.Session() as sess:
            sess.run(self.init) # initialization 
                
            # Restore model weights from previously saved model
            self.saver.restore(sess, self.model_path)
            #save_path = self.saver.save(sess, self.model_path)
            #print("Model restored from file: %s" % save_path) 
            
            prediction=tf.argmax(self.pred,1)
            RESULT = prediction.eval(feed_dict={self.x: x_HW}       ) 
            print("=================My hand written programme======================")
            print("Prediction :" ,  RESULT[0] ) 
            print("=================My hand written programme======================")
        return  RESULT[0]
 
#NotFoundError: Restoring from checkpoint failed. 
#This is most likely due to a Variable name or other graph key that is missing
# from the checkpoint. Please ensure that you have not altered the graph
# expected based on the checkpoint. Original error:
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


 
class ConvolutionLayer() :
    def __init__(self) : 
        tf.reset_default_graph()
        # Parameters
        self.learning_rate = 0.001
        self.batch_size = 100
        self.display_step = 1
        self.dropout= 1
        self.epoch = 20
        # create a subdirectory to save/restore the data  
        self.name_your_save_file="modelCNN"
        self.model_path =os.getcwd()+'\\CNN_data\\'+str(self.name_your_save_file) 
        
        
        # Network Parameters (CNN)
        self.CNN_grid_size=4
        self.CNN_input_dimension=1   #   due to monochrome. If input is RGB, this parameter needed to be set as 3. 
        self.CNN_filter_amount_1=8
        self.CNN_filter_amount_2=4
        # Network Parameters (FNN)
        self.n_hidden_0 = 49   # CNN (28x28)-> "2x2" max_pool ->  "2x2" max_pool -> FNN (7x7)
        self.n_hidden_1 = 98   # Amount of neurals in 1st (hidden) layer  
        self.n_hidden_2 = 196  # Amount of neurals in 2nd (hidden) layer  

        self.n_input = 784  # MNIST data input (img shape: 28*28)
        self.n_classes = 10 # MNIST total classes (0-9 digits)

        # tf Graph input
        self.x = tf.placeholder("float", [None, self.n_input ])
        self.y = tf.placeholder("float", [None, self.n_classes])

        # Store layers weight & bias
        self.weights = {
        
            'cn1': tf.Variable(tf.random_normal([self.CNN_grid_size, self.CNN_grid_size, self.CNN_input_dimension, self.CNN_filter_amount_1])),
            'cn2': tf.Variable(tf.random_normal([self.CNN_grid_size, self.CNN_grid_size, self.CNN_filter_amount_1, self.CNN_filter_amount_2])),
            'h1': tf.Variable(tf.random_normal([self.n_hidden_0*self.CNN_filter_amount_2, self.n_hidden_1])),
            'h2': tf.Variable(tf.random_normal([self.n_hidden_1, self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_hidden_2, self.n_classes]))
        }
        self.biases = {
            'cn1': tf.Variable(tf.random_normal([self.CNN_filter_amount_1])),
            'cn2': tf.Variable(tf.random_normal([self.CNN_filter_amount_2])),
            'b1': tf.Variable(tf.random_normal([self.n_hidden_1])),
            'b2': tf.Variable(tf.random_normal([self.n_hidden_2])),
            'out': tf.Variable(tf.random_normal([self.n_classes]))
        }

        # Construct model
        self.keep_prob = tf.placeholder(tf.float32) # dropout (keep probability)
        self.pred = self.multilayer_perceptron(self.x, self.weights, self.biases, self.keep_prob)
        self.output_softmax = tf.nn.softmax(self.pred)
        # Define loss and optimizer
        self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.pred, labels=self.y))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)

        # Initialize the variables (i.e. assign their default value)
        self.init = tf.global_variables_initializer()

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()


 
    # Create model
    def multilayer_perceptron(self , x, weights, biases, dropout):
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


 
    def train(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #test_X=  mnist.test.images.reshape(-1,28,28,1)
        #train_X= mnist.train.images.reshape(-1,28,28,1)
        with tf.Session() as sess:   
            # Run the initializer
            sess.run(self.init)        
            # Training cycle       
            for epoch in range(self.epoch):
                avg_cost = 0.
                total_batch = int(mnist.train.num_examples/self.batch_size)
                if improt_tqdm == True :  pbar = tqdm(total=total_batch+1)
                # Loop over all batches
                for i in range(total_batch):
                    if i%50 ==0 and improt_tqdm == True : pbar.update(50)
                    if i%50 ==0 and improt_tqdm == False : print("completed : " ,i,"/", total_batch)
                    batch_x, batch_y = mnist.train.next_batch(self.batch_size) 
                    # Run optimization op (backprop) and cost op (to get loss value)
                    _, c = sess.run([self.optimizer, self.cost], feed_dict={self.x: batch_x,self.y: batch_y, self.keep_prob : 1.0})
    
                    # Compute average loss
                    avg_cost += c / total_batch
                # Display logs per epoch step
                if improt_tqdm == True :  pbar.close()
                if epoch % self.display_step == 0:
                    print("Epoch:", '%04d' % (epoch+1), "cost=", \
                        "{:.9f}".format(avg_cost))
            print("Optimization Finished!")
        
            # Test model
            correct_prediction = tf.equal(tf.argmax(self.pred, 1), tf.argmax(self.y, 1))
     
            # Calculate accuracy
            accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
            print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels , self.keep_prob : 1.0}))
            
            # Save model weights to disk
            save_path = self.saver.save(sess, self.model_path)
            print("Model saved in file: %s" % save_path)

    def predict_mnist(self):
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        #test_X=  mnist.test.images.reshape(-1,28,28,1)
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
            print("Accuracy:", accuracy.eval({self.x: mnist.test.images, self.y: mnist.test.labels , self.keep_prob : 1.0}))
            print("The first 10 digits in the minit.test : ")
            print("Prediction :" ,prediction.eval(feed_dict={self.x: x_test ,self.keep_prob : 1.0})        )
            print("Labels     :"  ,sess.run(tf.argmax(y_test, 1))               )
            print("=============================================================")
        

    def predict_test(self):
        x_HW = load('test.json')
        x_HW=x_HW.reshape(-1,784)

        with tf.Session() as sess:
            sess.run(self.init) # initialization 
        
            # Restore model weights from previously saved model
            self.saver.restore(sess, self.model_path)
            save_path = self.saver.save(sess, self.model_path)
            print("Model restored from file: %s" % save_path) 
           
            prediction=tf.argmax(self.pred,1)
            RESULT = prediction.eval(feed_dict={self.x: x_HW,self.keep_prob : 1.0}      )       
            print("=================My hand written programme======================")
            print("Prediction :" ,RESULT[0]       )
            print("=================My hand written programme======================")
        
        return  RESULT[0] 
 
        
    
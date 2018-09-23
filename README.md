# Machine Learning 
#### python 3.5 & tensorflow
The place to store my toy models for ML. 
Most of these codes are constructed within the framework of Tensorflow.
For visualization, some additional packages are used. (such as pygame.) 

Notice : Some of these codes are not well organized and still under development.

Currently, this repository includes:


## a. MNIST handwritten digit recognition project ver2.  (9/3/2018 - 9/13/2018)

**keywords : Convolution Neural Network (CNN), Fully Connected Layer, writing panel**

#### Description : 

The basic neural network framework is based on [the work of Aymeric Damien](https://github.com/aymericdamien/TensorFlow-Examples/)

Training your own neural network, saving and restoring the tf.data for later use/retrain.
Fully Connected Layer and Convolution Layer are implemented in this project.
A writing panel is built (using pygame), users can interact with their own neural network directly. 

#### results & future work: 

Although both neural network can perform marvelously well in MNIST data, in terms of cost-performance ratio
(can easily achieve accuracy > 95% within a few minute training).
For both, the ability of recognition for the handwritten digits on writing panel are quite low-level.
To improve this, my assumptions is:
- Need to build a larger neural network, more and more compulational costs are required.
- The handwritten digits on writing panel maybe quite different from the **real** handwritten digits in the MINST database. 
~~My handwritten digits are nothing but a scrawl as my elementary school teacher always told me.~~



## b. seq2seq model (magica madoka)   (9/23/2018 -)

**keywords : Recurrent Neural Network (RNN), bidirectional RNN (bi-RNN) , long-short term memory (LSTM)**

#### Description : 

This work is inspired by [Ematvey's awesome tutorials](https://github.com/ematvey/tensorflow-seq2seq-tutorials) for seq2seq!

Full script of madoka magica. 
[魔法少女まどか☆マギカ台本](https://www22.atwiki.jp/madoka-magica/pages/83.html)

The basic framework is biRNN (LSTM).
In this model, the word vector is also trainable.  (Can be view as another full connected layer)
Feed all the conversation in madoka magica into seq2seq model.
Details are preparing ...
Training time : ~ 12 hour 


Caution : python dict and set as **hash** method, which can speed up the searching process while it also means to random generate a index for each word so the order of items in dictionary will be different if we restart the Console. 
So in encoder-decoder process, one must to notice that we use the same dictionary (it would be better to save the dictionary in .json or something else)

#### results & future work: 

The ability to recognition the words with similar meaning are very weak, which is quite reasonable since I only feed this 1 MB or so .txt file for training.

Using Work2Vec and larger data.

I am ready to delve into analyzing  (>200 MB) txt data  ...

For larger conversation : Attention model are needed.

Personalized word vector project : After 



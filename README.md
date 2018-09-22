# Machine Learning 
#### python 3.5 & tensorflow
The place to store my toy models for ML. 
Most of these codes are constructed within the framework of Tensorflow.
For visualization, some additional packages are used. (such as pygame.) 

Notice : Some of these codes are not well organized and still under development.

Currently, this repository includes:


## a. MNIST handwritten digit recognition project ver2.  (9/3/2018 - 18/3/2018)

**keywords : Convolution Neural Network (CNN), Fully Connected Layer, writing panel**

#### Description : 

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
- My handwritten digits are nothing but a scrawl as my elementary school teacher always told me.



## b. seq2seq model (magica madoka)   (9/23/2018 -)

**keywords : Recurrent Neural Network (RNN), bidirectional RNN (bi-RNN) , long-short term memory (LSTM)**

#### Description : 

Feed all the conversation in madoka magica into seq2seq model.

Training time : < 12 hour 

#### results & future work: 

The ability to recognition the words with similar meaning are very weak, which is quite reasonable since I only feed this 1 MB or so .txt file for training.

Using Work2Vec and larger data.
Attention model!

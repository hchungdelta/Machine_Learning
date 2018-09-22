import numpy as np
import tensorflow as tf
from tensorflow.contrib.rnn import LSTMCell, LSTMStateTuple
import read_txt as readtxt
import os
import sys
import json
 
tf.reset_default_graph()
PAD = 0  # means nothing 
EOS = 1  # start/end the sequence   


def decode(_index):
    if type(_index) == np.ndarray :
        _index= _index.reshape(-1)
    txt = ''
    for index in  _index  :
        if reverse_dictionary[index] != 'PAD' :
            txt += str(reverse_dictionary[index])
            if reverse_dictionary[index] == 'EOS' :
                txt +=(" ")
    return txt
def encode(_index):
    if type(_index) == list :
        pass
    else :
        _index= list(_index)
    output_encode = []
    for a in _index:
        try :
            output_encode.append(dictionary[a])          # if in dictionary
        except :
            output_encode.append(dictionary["UNKNOWN"])  # if not in dictionary
    return output_encode
def spec_vocab_searcher(_string):
    if type(_string) == str :
        list_input_phase = list(_string)
    pharse_include_spec_vocab = []
    char_num=0
    while char_num  < (len( list_input_phase) ) :
        match = False
        for vocab in spec_vocab :
            try :
                if list_input_phase[char_num] in vocab[0]  and match == False:
                    txt = ''
                    for length in range( len(vocab) ):
                         txt+= list_input_phase[char_num+length]
                    if txt == vocab  :
                        pharse_include_spec_vocab +=[txt]
                        match =True      
                        char_num += (len(txt))
            except :
                pass
        if match == False :            
            pharse_include_spec_vocab +=list_input_phase[char_num]
            char_num += 1
    return pharse_include_spec_vocab 
     


### read_txt ###
training_file ='madoka_short_conversation.txt'
training_data = readtxt.read_japanese(training_file)     #be typed as np.ndarray
data_input_phrase= readtxt.separate_into_phrase(training_data)

spec_vocab_txt = 'dictionary_thres_5.txt'
spec_vocab = readtxt.read_japanese_split_punction(spec_vocab_txt)  #be typed as np.ndarray

with open("conversation.json","r", encoding='utf-8') as jsonfile:
    F=json.load(jsonfile)
    data_input_phrase_spec_vocab =F['conversation']  
with open("dict.json","r", encoding='utf-8') as jsonfile:
    F=json.load(jsonfile)
    dictionary = F['dictionary'] 
    _reverse_dictionary = F['reverse_dictionary']  # the keys are "string"

reverse_dictionary= {} 
for key, value in _reverse_dictionary.items():  
    reverse_dictionary[int(key)] = value
 
vocab_size = len(dictionary)
input_word_uplimit =14   # it will cut off if the input sentence too long
input_embedding_size = 300
encoder_hidden_units = 100
decoder_hidden_units = encoder_hidden_units * 2 # due to bidirection RNN!
 


reshape_data_input_index=[]
for _index in  data_input_phrase_spec_vocab :
    _index=encode(_index)
    _index=readtxt.re_shape_the_phrase(_index,input_word_uplimit) 
    if len(_index) == input_word_uplimit :
        _index_reshape=np.reshape(_index,(1,-1,1))
        if  len(reshape_data_input_index)  == 0 :
            reshape_data_input_index = _index_reshape
        else :
            reshape_data_input_index= np.append(reshape_data_input_index,_index_reshape , axis=0)
        for b in _index :    
            if reverse_dictionary[b] != 'PAD' :
                if reverse_dictionary[b] == 'EOS' :
                    pass
encoder_inputs_data  = reshape_data_input_index     # phrase 0 1 2 3 4 5 ...
decoder_targets_data = reshape_data_input_index[1:] # phrase 1 2 3 4 5 6 ...



print("input data ... completed!")


## Neural network ##
encoder_inputs        = tf.placeholder(shape=(None, None), dtype=tf.int32, name='encoder_inputs')
encoder_inputs_length = tf.placeholder(shape=(None,), dtype=tf.int32, name='encoder_inputs_length')
decoder_targets       = tf.placeholder(shape=(None, None), dtype=tf.int32, name='decoder_targets')
embeddings            = tf.Variable(tf.random_uniform([vocab_size, input_embedding_size], -1.0, 1.0), dtype=tf.float32)

#embeddings can be viewed as word vector repository. one row represents one certain word's vector. 
encoder_inputs_embedded = tf.nn.embedding_lookup(embeddings, encoder_inputs)


encoder_cell = LSTMCell(encoder_hidden_units)
(encoder_fw_outputs,encoder_bw_outputs),(encoder_fw_final_state,encoder_bw_final_state) =\
 tf.nn.bidirectional_dynamic_rnn(   cell_fw=encoder_cell,
                                    cell_bw=encoder_cell,
                                    inputs=encoder_inputs_embedded,
                                    sequence_length=encoder_inputs_length,
                                    dtype=tf.float32, time_major=True)

encoder_outputs = tf.concat((encoder_fw_outputs, encoder_bw_outputs), 2)  #concatenate (?(time-steps),?(batch),20) --> (?,?,40)

encoder_final_state_c = tf.concat((encoder_fw_final_state.c, encoder_bw_final_state.c), 1)
encoder_final_state_h = tf.concat((encoder_fw_final_state.h, encoder_bw_final_state.h), 1)
#encoder_final_state.h : activations of hidden layer of LSTM cell
#encoder_final_state.c : final output, which can potentially be transfromed with some wrapper 
encoder_final_state   = LSTMStateTuple(c=encoder_final_state_c,h=encoder_final_state_h) 

decoder_cell                 = LSTMCell(decoder_hidden_units) 
encoder_max_time, batch_size = tf.unstack(tf.shape(encoder_inputs))
 
decoder_lengths = encoder_inputs_length #+ 3


W = tf.Variable(tf.random_uniform([decoder_hidden_units, vocab_size], -1, 1), dtype=tf.float32)
b = tf.Variable(tf.zeros([vocab_size]), dtype=tf.float32)



assert EOS == 1 and PAD == 0
 
eos_time_slice    = tf.ones([batch_size], dtype=tf.int32, name='EOS')
pad_time_slice    = tf.zeros([batch_size], dtype=tf.int32, name='PAD')
eos_step_embedded = tf.nn.embedding_lookup(embeddings, eos_time_slice)   
pad_step_embedded = tf.nn.embedding_lookup(embeddings, pad_time_slice)



def loop_fn_initial():
    initial_elements_finished = (0 >= decoder_lengths)  # all False at the initial step
    initial_input = eos_step_embedded   # All EOS in the size "batch"
    initial_cell_state = encoder_final_state
    initial_cell_output = None
    initial_loop_state = None  # we don't need to pass any additional information
    return (initial_elements_finished,
            initial_input,
            initial_cell_state,
            initial_cell_output,
            initial_loop_state)

def loop_fn_transition(time, previous_output, previous_state, previous_loop_state):
    def get_next_input():
        output_logits = tf.add(tf.matmul(previous_output, W), b)
        prediction = tf.argmax(output_logits, axis=1)
        next_input = tf.nn.embedding_lookup(embeddings, prediction)
        return next_input
    
    elements_finished = (time >= decoder_lengths) # this operation produces boolean tensor of [batch_size]
                                                  # defining if corresponding sequence has ended

    finished = tf.reduce_all(elements_finished) # -> boolean scalar (if False exists : False)
    input = tf.cond(finished, lambda: pad_step_embedded, get_next_input)
    state = previous_state
    output = previous_output
    loop_state = None
    return (elements_finished, input,state,output,loop_state)


def loop_fn(time, previous_output, previous_state, previous_loop_state):
    if previous_state is None:    # time == 0
        assert previous_output is None and previous_state is None
        return loop_fn_initial()
    else:
        return loop_fn_transition(time, previous_output, previous_state, previous_loop_state)

decoder_outputs_ta, decoder_final_state, _ = tf.nn.raw_rnn(decoder_cell, loop_fn)
decoder_outputs= decoder_outputs_ta.stack()
decoder_max_steps, decoder_batch_size, decoder_dim = tf.unstack(tf.shape(decoder_outputs))
decoder_outputs_flat = tf.reshape(decoder_outputs, (-1, decoder_dim))

decoder_logits_flat = tf.add(tf.matmul(decoder_outputs_flat, W), b)
decoder_logits = tf.reshape(decoder_logits_flat, (decoder_max_steps, decoder_batch_size, vocab_size))
decoder_prediction = tf.argmax(decoder_logits, 2)


 


stepwise_cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
    labels=tf.one_hot(decoder_targets, depth=vocab_size, dtype=tf.float32),
    logits=decoder_logits)

loss = tf.reduce_mean(stepwise_cross_entropy)
train_op = tf.train.AdamOptimizer(learning_rate=0.0005).minimize(loss)


batch_size = 1

def next_feed():
    global next_feed_run_time, amount_of_loop_in_txt
    encoder_inputs_ = encoder_inputs_data[next_feed_run_time]
    decoder_targets_ =decoder_targets_data[next_feed_run_time] 
    next_feed_run_time +=1      
    if next_feed_run_time > len(decoder_targets_data) -1:
        next_feed_run_time = 0
        amount_of_loop_in_txt +=1
        
    encoder_input_lengths_ = [input_word_uplimit]
    return {encoder_inputs: encoder_inputs_,
            encoder_inputs_length: encoder_input_lengths_,
            decoder_targets: decoder_targets_}

amount_of_loop_in_txt=0
amount_of_loop_in_txt_uplimit=150
next_feed_run_time=0
loss_track = []
display_result_interval = 1000

saver = tf.train.Saver()
model_path= os.getcwd()+'\\DataSave\\'+str('madoka_training') 
WORK=False
keep_on_train =True
CHECK =True
 
if WORK :
    with tf.Session() as sess:
        if keep_on_train == True :
            saver.restore(sess, model_path)
            save_path = saver.save(sess, model_path)
            print("Model keep on training!!") 
        else :
            sess.run(tf.global_variables_initializer())
        training_step = 0
        try:
            while amount_of_loop_in_txt < amount_of_loop_in_txt_uplimit :
                
                training_step +=1
                fd = next_feed()
                _, l = sess.run([train_op, loss], fd)
                loss_track.append(l)
                if  training_step % display_result_interval == 0:
                    print('Loop {}/{}  ,training_step {}'.format(
                            amount_of_loop_in_txt,amount_of_loop_in_txt_uplimit,training_step))
                    print('  minibatch loss: {}'.format(sess.run(loss, fd)))
                    predict_ = sess.run(decoder_prediction, fd)
                    
                    for i, (inp, pred) in enumerate(zip(fd[encoder_inputs].T, predict_.T)):
                        print('  sample {}:'.format(i + 1))
                        print('  Input  :',decode(inp) )
                        print('  Output :',decode(pred) )
                        if i >= 2:
                            break
                    print()
                    save_path = saver.save(sess, model_path)                     
                    print("data saved")
        
        except KeyboardInterrupt:
            print('training interrupted')



if CHECK: 
     #with tf.Session(config=tf.ConfigProto(intra_op_parallelism_threads=1) ) as sess:
     with tf.Session(  ) as sess:

         #sess.run(init) # initialization 
        # Restore model weights from previously saved model
        saver.restore(sess, model_path)
        while True :
            keyin=input("input :")
            if keyin == "exit()":
                sys.exit()
            try :
                keyin_spilt_keyword=spec_vocab_searcher(keyin)
                test = readtxt.re_shape_the_phrase(encode(keyin_spilt_keyword),input_word_uplimit)
                print(test)
                test = np.array([test]) # need to change to np.array
                test = test.reshape(-1,1) 
                predict_ = sess.run(decoder_prediction, feed_dict={encoder_inputs:  test,  encoder_inputs_length:[input_word_uplimit] })
                for i,  (inp, pred) in enumerate(zip(test.T, predict_.T)):
                    print('  Input  :',decode(inp) )
                    print(pred)
                    print('  Output :',decode(pred) )
            except :
                print("something wrong")
 
                

     
 
    
 
 
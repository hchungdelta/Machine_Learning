"perpare data"
import numpy as np
import read_txt as readtxt
import json

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
     

training_file ='madoka_short_conversation.txt'
training_data = readtxt.read_japanese(training_file)     #be typed as np.ndarray
data_input_phrase= readtxt.separate_into_phrase(training_data)

spec_vocab_txt = 'dictionary_thres_5.txt'
spec_vocab = readtxt.read_japanese_split_punction(spec_vocab_txt)  #be typed as np.ndarray



data_input_phrase_spec_vocab=[]
for a in data_input_phrase :
    new_pharse=spec_vocab_searcher(a)
    data_input_phrase_spec_vocab.append(new_pharse)        
data_input_spec_vocab=[word  for single_phrase in data_input_phrase_spec_vocab 
                       for word in single_phrase]
conver={'conversation': data_input_phrase_spec_vocab}

with open("conversation.json","w", encoding='utf-8') as jsonfile:
    json.dump(conver,jsonfile,ensure_ascii=False)

'read part in main.py'
'''
with open("conversation.json","r", encoding='utf-8') as jsonfile:
    F=json.load(jsonfile)
    data_input_phrase_spec_vocab =F['conversation']  
 '''
    

dictionary, reverse_dictionary = readtxt.build_dataset(data_input_spec_vocab  )
exDict = {'dictionary': dictionary,'reverse_dictionary': reverse_dictionary}

with open("dict.json","w", encoding='utf-8') as jsonfile:
    json.dump(exDict,jsonfile,ensure_ascii=False)

'read part in main.py'
'''
with open("dict.json","r", encoding='utf-8') as jsonfile:
    F=json.load(jsonfile)
    dictionary = F['dictionary'] 
    _reverse_dictionary = F['reverse_dictionary']  # the keys are "string"

reverse_dictionary= {} 
for key, value in _reverse_dictionary.items():  
    reverse_dictionary[int(key)] = value
'''







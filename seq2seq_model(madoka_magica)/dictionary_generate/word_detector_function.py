import collections  
import os 
import json


def txt_data_input(fname, up_limit_char=5):
    with open(fname, encoding = 'utf-8') as f:
        txt_file=f.readlines()        
    word_dict=[]
    # create dict
    for _ in range(up_limit_char+1):
        word_dict.append ( dict()   )
    # create word repository
    for sentence in txt_file:
        sentence_tolist=list(sentence)[:-1]   # not to include \n 
        for b in  range(len(sentence_tolist) ):
            try :
                if  sentence_tolist[b] not in word_dict[1].keys() :  # create a new item if not exist in advance.
                    for num in range(1,up_limit_char+1):
                        word_dict[num][sentence_tolist[b]] =  [] 
                candidate_word = '' # begin to record the word, for example,
                                    # If the the first word is "魔", 
                                    # it will record "魔法" ,"魔法少" ... and "魔法少女"
                                    # if the up_limit_char is set to be 4.
                for num in range(1,up_limit_char+1):  
                    candidate_word += str(sentence_tolist[b+num-1] )
                    word_dict[num][sentence_tolist[b]].append(candidate_word )  
            except : 
                pass    
    return word_dict


def vocab_filter(word_dict,threshold_for_first_word,threshold_for_record,up_limit_char, broke_down=False ):
    filtered_vocab  =''
    dict_type=dict()
    for a in word_dict[1] :     
        if len(word_dict[1][a]) > threshold_for_first_word  :    
            word_counter = [] 
            for b in  range(up_limit_char+1)  :
                try :
                    word_counter.append(collections.Counter(word_dict[b][a]).most_common(20) )
                except :
                    word_counter.append(dict())
            for c in  reversed( range(1,up_limit_char+1) ) :
                if c == 1  : 
                    for ( key , value ) in word_counter[c] :
                        if    value > threshold_for_record :    
                            filtered_vocab +=str(c)+str(':')+str(key)+str(':')+str(value) +str('\n')
                            dict_type[key]=value
                if c > 1  :                 
                    for ( key , value ) in word_counter[c] :
                        if  value > threshold_for_record:
                            filtered_vocab +=str(c)+str(':')+str(key)+str(':')+str(value) +str('\n')
                            dict_type[key]=value
                            if broke_down == True:
                                for layer in range(1,c):
                                    for  index , ( key_pre , value_pre ) in  enumerate ( word_counter[c-layer] ):
                                        if key[:(c-layer)] == key_pre:
                                            word_counter[c-layer][index] = (key_pre , value_pre- value )    
    return  filtered_vocab , dict_type
def threshold_for_raw_dictionary(jsonfile,threshold):
    with open(jsonfile,"r", encoding='utf-8') as jsonfile:
        unrefined_dict=json.load(jsonfile)["vocab_dict"]
    unrefined_dict_pre1=dict()
    for key, value in unrefined_dict.items() :
        if int(value) > threshold  :
            #print("length : %d   amount %4d  word : %s" %(len(key) , value,key ) )
            unrefined_dict_pre1[key]=[len(key) , value]
    return (unrefined_dict_pre1,len(unrefined_dict_pre1))
#########################################################
'''
txt_target_file = os.getcwd()+"\\sister\\sister7_split.txt"
up_limit_char=6
word_dict = txt_data_input(txt_target_file,up_limit_char)
 
threshold_for_record = 1    # lower than this value, the potential vocab will be discarded. 
threshold_for_first_word   = 0 # create a term if the first word appears in the txt more than N time.
filtered_vocab= vocab_filter(threshold_for_first_word,threshold_for_record,up_limit_char)
#print { k: x.get(k, 0) + y.get(k, 0) for k in set(x) | set(y) }
   
with open("pre_dictionary.txt", "w" , encoding = 'utf-8') as file:
    file.write(filtered_vocab)                                 
'''
# There are some high frequency words turn out to be nonsense.
# for exmaple, 
#  word : frequence
# キュゥべえ : 368
# ュゥべえ : 368
# ゥべえ : 368  
'''
training_file = 'pre_dictionary.txt'
data=read_japanese_line(training_file)
data_done=read_japanese_line(training_file)  # copy for later revision
item_to_be_omitted=[]


for layer in reversed ( range(2,up_limit_char) ):
    two_char_list=[]
    
    for j ,a in enumerate (data):
        a=(a[:-1])  # omit \n
        a = a.split(':')
        if int(a[0]) == layer :
            two_char_list.append( (j ,a[1],a[2] )  )
    
    for i,a in enumerate (data):
        a=(a[:-1])  # omit \n
        a = a.split(':')
        if int(a[0]) > layer :
            for j , two_char, value in two_char_list:
                if two_char in str(a[1]):    
                    #print (  a[1] , 'from ', two_char ,':', int(a[2]),'to',value )
                    if  float(value)*0.9  <= float(a[2]) <= float(value)*1.1 :
                        print ( j, a[1] , 'from ', two_char ,':', int(a[2]),'to',value )
                        item_to_be_omitted.append(j)
                    else :
                        pass



item_to_be_omitted = sorted(item_to_be_omitted)
dictionary='' 
for i,a in  enumerate(data_done) :
    a=a.split(':')
    if ( int (a[0]) ) != 1 :    
        if i not in item_to_be_omitted :        
             dictionary+=str(a[1])+str(',')

with open("dictionary_thres_5.txt", "w" , encoding = 'utf-8') as file:
    file.write(dictionary)     
# There are some very high frequency terms like の　を　が　etc ...
# and these terms tend to adhere to the word in front or after a certain word,
# it can form some weird word like ... の子 の人
# well, it seems innocuous currently... ^.^

'''

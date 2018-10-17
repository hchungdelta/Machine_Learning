import os
import json
import word_detector_function as WDF
 

raw_vocab_dict=dict()
for num in range(1,11):
    
    txt_target_file = os.getcwd()+u"/sister%d/total_split.txt"  %num
    up_limit_char   = 7            # the up limit length for the word , e.g. how many characters it can have.
    word_dict = WDF.txt_data_input(txt_target_file,up_limit_char)


    up_limit_char   = 7            # the up limit length for the word , e.g. how many characters it can have.
    threshold_for_record       = 3 # lower than this value, the potential vocab will be discarded.
    threshold_for_first_word   = 0 # create a term if the first word appears in the txt more than N time.
    filtered_vocab, new_dict_input= WDF.vocab_filter( word_dict, threshold_for_first_word,threshold_for_record,up_limit_char)
    raw_vocab_dict={item: raw_vocab_dict.get(item, 0) + new_dict_input.get(item, 0) for item in set(raw_vocab_dict) | set(new_dict_input)  }
    print("process :" , num , "-- completed!!")


data = {"vocab_dict" : raw_vocab_dict}
with open("raw_vocab_dict.json","w", encoding='utf-8') as jsonfile:
    json.dump(data,jsonfile,ensure_ascii=False)


input_json_file="raw_vocab_dict.json"
threshold =20
dict_after_threshold , amount_after_threshold=WDF.threshold_for_raw_dictionary("raw_vocab_dict.json",threshold)
 



#    for key, value in unrefined_dict.items() :
#        if "保持" in key:
#            if value > 1 :
#                print(key,value)

dele=0
up_limit_char=7
tolerance=0.1
item_to_be_omitted=set()
for layer in reversed ( range(2,up_limit_char) ):
    two_char_list=[]
    index=0
    for    key , ( len_of_word ,value) in  dict_after_threshold.items()  :
 
        if  len_of_word  == layer :
            two_char_list.append( (index ,key, value )  )
            index +=1
        

    for key , ( len_of_word,value) in dict_after_threshold.items():
        if len_of_word  > layer :
            for (index , short_key , short_key_value) in two_char_list:
                #print(short_key)
                if short_key  in  key :
                    if  float(value)*(1-tolerance) <= float(short_key_value) <= float(value)*(1+tolerance) :
                        print ( index , key, 'will replace ', short_key ,':', value,'to',short_key_value )
                        dele+=1
                        item_to_be_omitted.update([short_key])
                    else :
                        pass

print("amount of word",amount_after_threshold)
print("it will delete the amount of  ",dele)

dictionary='' 
vocab_dict=[]
vocab_amount_dict=dict()
for   key , ( len_of_word ,value) in  dict_after_threshold.items()  :
    if ( len_of_word  ) != 1 :    
        if key not in item_to_be_omitted :   
            dictionary += "%s : %d \n" %(key,value)
            vocab_dict.append(key)
            vocab_amount_dict[key]=value
vocab_dict={"vocab_dict": vocab_dict}
vocab_amount_dict={"vocab_amount_dict": vocab_amount_dict}
with open("DICT_NANA.txt", "w" , encoding = 'utf-8') as file:
    file.write(dictionary)
data = {"vocab_dict" : vocab_dict}
with open("vocab_dict_NANA.json","w", encoding='utf-8') as jsonfile:
    json.dump(vocab_dict,jsonfile,ensure_ascii=False) 

with open("vocab_amount_dict_NANA.json","w", encoding='utf-8') as jsonfile:
    json.dump(vocab_amount_dict,jsonfile,ensure_ascii=False)




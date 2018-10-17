import json
with open("DICT_NANA.txt","r",encoding='utf-8') as txtfile:
    txt=txtfile.readlines()

with open("vocab_amount_dict_NANA.json","r", encoding='utf-8') as jsonfile :
    dictionary=json.load(jsonfile)["vocab_amount_dict"]
new_dict_txt=""
new_dict_list=[]
for line in txt:
    line=line[:-1]
    delete_this_line = False

    if len(line.split(":")) == 2 :
        key , value=line.split(":")
    else : 
        continue
    value=int(value)
    key=key[:-1]
    if len(key) > 3 :
        print( key ,end=' ')
        
        print(dictionary[key]) 
        char=list(key)
        pair_word=[]
        MAXscore=0
        for length in range(2,len(key)-1) :
            first_word="".join(char[:length]) 
            second_word="".join(char[length:])
            
            try :
                #print("--", first_word , end=' ')
                #print(dictionary[first_word]   )
                #print("--",second_word , end=' ')
                #print(dictionary[second_word]   )

                score=dictionary[first_word]*dictionary[second_word] 
                if score > MAXscore and score > 3*value*value:
                    MAXscore=score
                    pair_word=[ first_word,second_word]
                    delete_this_line = True
            except :
                pass
                print(  "Not in dict")
        print( pair_word, "can replace you!!!!")
        
    if delete_this_line == False and value > 65 :
        new_dict_txt += line +'\n'
        new_dict_list.append(key)


vocab_dict={"vocab_dict": new_dict_list}

with open("newDICTNANA.txt",'w',encoding='utf-8') as  txtfile :
    txtfile.write(new_dict_txt)
with open("DICT_NANA_opti.json","w",encoding='utf-8') as jsonfile :
    
    json.dump(vocab_dict,jsonfile,ensure_ascii=False)


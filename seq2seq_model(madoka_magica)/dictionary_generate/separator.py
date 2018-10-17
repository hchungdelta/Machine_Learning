def separator(target_txt):
    if target_txt.endswith("_split.txt"):
        print (target_txt+ " is already separated !")
        print("Separator will not separate the _split.txt file!")
        return 
    # save a new file,  for example example.txt --> example_split.txt
    save_name="".join(list(target_txt)[:-4])+str("_split.txt")    
    with open(target_txt, encoding = 'utf-8') as f:
        content =f.readlines()         
    Punctuation='「？！…。」～、（『,・』!?. 【】―&;）[]＝》《　 '
    sub_sentence_list = []
    record=True
    write_into_txt = ''
    
    for sentence in  content :
        
        sentence_tolist=list(sentence)
        sub_sentence = ''
        for  (i,char) in enumerate(sentence_tolist):
            if char == '(' :
                record=False
            if record :
                if char in Punctuation :
                    #print (sub_sentence)    
                    sub_sentence_list.append(sub_sentence)
                    write_into_txt +=sub_sentence +str('\n')
                    sub_sentence = ''
                
                else :
                    sub_sentence += char
            if char == ')':
                record=True
 
    with open(save_name, "w" , encoding = 'utf-8') as file:
        file.write(write_into_txt) 
 
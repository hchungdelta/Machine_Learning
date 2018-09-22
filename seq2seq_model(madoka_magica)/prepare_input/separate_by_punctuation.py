def read_japanese_line(fname):
    with open(fname, encoding = 'utf-8') as f:
        content =f.readlines()        
    return content### read_txt ###
training_file = 'madoka.txt'
data=read_japanese_line(training_file)

Punctuation='「？！…。」～、)(（『,・』!?.'
sub_sentence_list = []
write_into_txt = ''
for sentence in  data  :
    
    sentence_tolist=list(sentence)
    sub_sentence = ''
    for  (i,char) in enumerate(sentence_tolist):
        if char in Punctuation :
            #print (sub_sentence)    
            sub_sentence_list.append(sub_sentence)
            write_into_txt +=sub_sentence +str('\n')
            sub_sentence = ''
            
        else :
            sub_sentence += char
  
 
with open("separate_by_punctuation.txt", "w" , encoding = 'utf-8') as file:
    file.write(write_into_txt) 
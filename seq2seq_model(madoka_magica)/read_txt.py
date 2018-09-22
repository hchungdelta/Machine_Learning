 
import collections

def read_japanese(fname):
    with open(fname, encoding = 'utf-8') as f:
        content =f.read()
    content=list(str(content) )
    return content
def read_japanese_split_punction(fname):
    with open(fname, encoding = 'utf-8') as f:
        content =f.read()
    content=content.split(',')
    return sorted(content ,key=len,reverse=True) 
def build_dataset(words):
    count = collections.Counter(words).most_common()
    dictionary = dict()
    dictionary['PAD']=0
    dictionary['EOS']=1
    dictionary['UNKNOWN']=2
    for word, _ in count:
        dictionary[word] = len(dictionary) # give each word a number
        # just to sort words from high frequency to low
    reverse_dictionary = dict(zip(dictionary.values(), dictionary.keys()))
    return dictionary, reverse_dictionary

def separate_into_phrase_and_index(input_txt, dictionary):
    data_input_phrase=[]
    data_input_index=[]
    index_for_this_phrase=[]
    phrase=''
    num=0   
    for a in input_txt:
        if a == '\n' :
            num +=1 
            data_input_phrase.append(phrase)
            data_input_index.append(index_for_this_phrase)
            phrase=''
            index_for_this_phrase=[]
        else :
            phrase+=str(a)
            index_for_this_phrase.append(dictionary[a])
    return data_input_phrase,data_input_index 


def separate_into_phrase(input_txt):
    data_input_phrase=[]
    phrase=''
    num=0   
    for a in input_txt:
        if a == '\n' :
            num +=1 
            data_input_phrase.append(phrase)
            phrase=''
        else :
            phrase+=str(a)
    return data_input_phrase



def re_shape_the_phrase(_index,input_word_uplimit ):  
    _need_to_add=input_word_uplimit - len(_index)
    if _need_to_add <= 0 :
        _index[input_word_uplimit -1]=1
        del _index[input_word_uplimit : ]
    else : 
        _index.append(1)
        _need_to_add -=1        
    while _need_to_add > 0 :
        _need_to_add -=1
        _index.append(0)     
    return ( _index )
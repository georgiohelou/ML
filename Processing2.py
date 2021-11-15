from typing import Dict


def process_inverted():  
    with open('test2.txt') as f:
        lines = f.readlines()
        Dict = {}
        for i in range(len(lines)): # read paper by paper
            current_line=lines[i]
            strX = current_line
            key=int(current_line.split('----')[0]) # extract paper/abstract ID
            strX = strX.split('----')[1]
            abstract_size=current_line.split('\"IndexLength\":')[1]
            strX = strX.split('\"IndexLength\":')[1]
            abstract_size=int(abstract_size.split(",")[0]) # read IndexLength value
            strX = strX.split(',\"InvertedIndex\":{')[1]
            abstract_array=[None]*abstract_size
            if strX[-1] == "}" and strX[-2] == "}":
                strX = strX[:-2]  # rmv "}}" at the very end
                strX += '++++++'  # setting flag
            
            count = 0
            while strX.split('++++++')[0] is not '':
                strY = strX.split('\"', 2) # split based on 2 "                
                word = (strY[1])
                # word = process_word(word)
                strX = strY[2]
                strY = strX.split('[', 1)[1]
                strY = strY.split(']', 1)
                indices = strY[0]
                process_indices(word, indices, abstract_array)
                strX = strY[1]
                count += 1

            print(abstract_array)
            s = generate_abstract(abstract_array)
            print(s)
            return
            

def process_indices(word, x, arr):
    ## input: word and string x containing comma-separated indices for the word
    if len(x) == 0:
        print("no index... error")
    elif len(x) == 1:
        arr[int(x)] = word
    else:
        x = x.replace(" ", "") # remove spaces
        x_arr = x.split(",")
        for i in x_arr:
            arr[int(i)] = word
    return

def process_word(word):
    ## function cleans the word, removing bad characters
    word = ''.join(e for e in word if e.isalnum())
    return word

def generate_abstract(arr):
    s = ""
    for word in arr:
        s += " " + word
    return s 

process_inverted()

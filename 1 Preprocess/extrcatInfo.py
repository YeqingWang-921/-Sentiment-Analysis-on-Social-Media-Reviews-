import os
import re

if __name__=='__main__':
    path = 'F:\\aclImdb_v1\\pros-cons'
    filename = 'Cons_total.txt'
    outpath = 'Cons_ex.txt'

    # remove tab
    inputfile = open(path + '\\' + filename, encoding='utf-8')
    lines = inputfile.readlines()
    with open(path + '\\' + outpath, 'a+',encoding='utf-8')as f:
        for line in lines:
            print(line)
            #remove tab
            #sent = line.strip(' ')
            #print(sent)
            s = line[6:]
            news = s[:-8]
            sentence = news + '\n'
            f.write(sentence)
    f.close()


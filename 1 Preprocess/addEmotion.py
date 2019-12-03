import os
import re

if __name__=='__main__':
    '''
    pos: 1
    neg: -1
    '''
    path = 'F:\\aclImdb_v1\\IMDB'
    orifilepath = 'pos\\pos_check.txt'
    newfile = 'pos\\final_pos_emo.txt'
    file = open(path + '\\' + orifilepath, encoding='utf-8')
    lines = file.readlines()
    #create a new file contains emotion mark
    with open(path + '\\' + newfile, 'a+', encoding='utf-8') as f:
        for line in lines:
            '''
            #for LJY
            while '\n' in line:
                line = line.replace('\n', '')

            #general
            f.write(line + '    '+ '0' + '\n')
            '''
            f.write('1' + '\t' + line)


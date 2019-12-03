import os
import re
import nltk

if __name__=='__main__':
    '''
    divide the original dataset into train validation test, and the ratio is 8:1:1
    '''
    path = 'F:\\aclImdb_v1\\pros-cons'
    #path = 'F:\\aclImdb_v1\\IMDB\\LJY'
    orifile = 'tstPros.txt'
    #trainfile = 'trainCons.txt'
    valfile = 'valPros.txt'
    testfile = 'testPros.txt'

    raw_word_list = []
    sent_list = []
    with open(path + '\\' + orifile, encoding='utf-8') as f:
        line = f.readline()
        while line:
            if len(line) > 0:
                sentence = str(line)
                raw_word_list = list(nltk.word_tokenize(sentence))
            sent_list.append(raw_word_list)
            line = f.readline()


    #trf = open(path + '\\' + trainfile, 'a+' , encoding= 'utf-8')
    tesf = open(path + '\\' + testfile, 'a+' , encoding= 'utf-8')
    valf = open(path + '\\' + valfile, 'a+' , encoding= 'utf-8')
    count = 0
    for sent in sent_list:
        '''
        if count % 5 == 0:
            for i in range(len(sent)):
                s = sent[i] + ' '
                tesf.write(s)
            tesf.write('\n')
        else:
            for i in range(len(sent)):
                s = sent[i] + ' '
                trf.write(s)
            trf.write('\n')
        count += 1

        '''
        #divide test & validation
        if count % 2 == 0:
            for i in range(len(sent)):
                s = sent[i] + ' '
                tesf.write(s)
            tesf.write('\n')
        else:
            for i in range(len(sent)):
                s = sent[i] + ' '
                valf.write(s)
            valf.write('\n')
        count += 1



    f.close()
    tesf.close()
    #trf.close()
    valf.close()

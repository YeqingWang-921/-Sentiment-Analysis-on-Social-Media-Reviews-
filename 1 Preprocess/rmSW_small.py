# -*- coding:utf-8 -*-
import glob
import os
import re
import nltk

if __name__=='__main__':
    txt_filenames = glob.glob('F:\\aclImdb_v1\\aclImdb\\test\\pos\\*.txt')
    txt_te_pos = open("F:\\aclImdb_v1\\IMDB\\pos\\pos_total.txt", "a+", encoding='UTF-8')
    path = 'F:\\aclImdb_v1\\aclImdb'

    #initialize stopwords list
    stopWords= []
    spfilename = 'stopwords.txt'
    with open(path + '\\' + spfilename, encoding='utf-8') as f:
        line = f.readline()
        print('start read stopwords')
        while line:
            # while line is not null
            stopWords.append(line[:-1])
            # [:-1] reserve \n of every line in stop words
            line = f.readline()
            # remove repeat stop words
        stopWords = set(stopWords)
        print('read done')

    for filename in txt_filenames:
        #print(filename)
        raw_word_list = []
        sent_list = []
        with open(filename, encoding='utf-8') as f:
            line = f.readline()
            while '\n' in line:
                line = line.replace('\n', '')
            if re.findall(u'[!\d"#$%&\'()*+,./:;<=>?@^_`{|}~]', line):
                tmp = re.findall(u'[!\d"#$%&\'()*+,./:;<=>?@^_`{|}~]', line)
                for i in tmp:
                    line = line.replace(i,'')
            if len(line) > 0:
                sentence = str(line)
                print(sentence)
                raw_word_list = list(nltk.word_tokenize(sentence))
                dealed_words = []
                for raw_word in raw_word_list:
                    word = str(raw_word)
                    stra = 'and'
                    strt = 'too'
                    #print('original word: ', word)

                    #print(word)'''word not in stopWords and'''
                    if word not in stopWords and ( word != stra ) and ( word != strt ):

                        print('dealed word: ', word.lower())
                        #raw_word_list.append(word)
                        #change the word in lower case to reduce redundency
                        lower_word = word.lower()
                        dealed_words.append(lower_word)
                sent_list.append(dealed_words)
            #line = f.readline()
        for sent in sent_list:
            for i in range(len(sent)):
                s = sent[i] + ' '
                txt_te_pos.write(s)
            txt_te_pos.write('\n')



    print("done!")
    #txt_file.close()
    txt_te_pos.close()
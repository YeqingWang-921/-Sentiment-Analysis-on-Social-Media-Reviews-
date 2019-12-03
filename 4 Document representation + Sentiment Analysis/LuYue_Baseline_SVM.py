# -*- coding: utf-8 -*-
"""
Baseline SVM: 
    Train and test baseline BoW+SVM
    (1) Segment and clean labeled comments
    (2) Represent documents(comments) using CountVectorizer (based on word frequency)
    (3) Split train & test data
    (4) Train and test machine learning model (SVM)
    
@author: Yue, Lu
"""
import os
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from  sklearn.svm import SVC #support vector classifier
import nltk

def cleanLabeledComment(labeled_comment_path,labeled_comment_file):
    '''
    Function:
        Segment and clean labeled comments
        
    Parameters:
        labeled_comment_file: name of .txt labeled comment file (for each line: [label comment])
        labeled_comment_path: absolute path of file

    Return:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]
        sentence_list:  a list of comments  [[word1,word2,word3,...,wordn],[word1,word2,word3,...,wordn],...,[word1,word2,word3,...,wordn]]
        raw_word_list:  a list of all words [word1,word2,word3,...,wordm]
    '''

    
    # Doing the segmentation using NLTK
    raw_word_list   = []
    sentence_list   = []
    sentiment_list  = []
    

    label_comment=open(labeled_comment_path+'/'+labeled_comment_file,encoding='utf-8')
    
    for line in label_comment:
        while '\n' in line:
            line = line.replace('\n','')
    
                
        data_label=line.split('\t',1)
        
        #transfer utf-8 code
        if data_label[0]=='\ufeff1':
            data_label[0]='1'
        # remove all punctuation and number in comment
        # \d number
        # \w English char and number
        if re.findall(u'[^\da-zA-Z]+',data_label[1]):  
            tmp=re.findall(u'[^\da-zA-Z]+',data_label[1])
            for i in tmp:
                data_label[1]=data_label[1].replace(i,' ')        
        label=data_label[0]
        data=data_label[1]
        # only load labeled comments
        if label not in ['-1','0','1']:
            continue

        if len(data)>=2: 
        # if current line is not null or single word
            # do the segmentation using Jieba
            #raw_words = list(jieba.cut(data,cut_all=False))
            raw_words =nltk.word_tokenize(data)
            dealed_words = []
            for word in raw_words:                 
                if word not in ['www','com','http']:
                    raw_word_list.append(word)
                    # raw_word_list: all words in file [word1,word2,....,wordn]
                    dealed_words.append(word)
                    # dealed_words: words in current line
            # remove null list after clean text
            if(len(dealed_words)>=2 and label !='0'):
                sentence_list.append(dealed_words)
                sentiment_list.append(int(label))
 
    # (1) Calculate the average sentence length K
    sentLength=len(raw_word_list)/len(sentence_list)+5
    print(sentLength)

    # (2) Normalize the length of sentence:
    #       length of comment > average sentence length => extract key words using TF-IDF
    #       length of comment < average sentence length => 0-padding
    for i in range(len(sentence_list)):
        if len(sentence_list[i])>sentLength:
            #sentence_list[i]=jieba.analyse.extract_tags(str(sentence_list[i]),topK=int(sentLength+1))
            sentence_list[i]=sentence_list[i][:int(sentLength+1)]
            while(len(sentence_list[i])<sentLength+1):
            #while(len(sentence_list[i])<sentLength):
                sentence_list[i].append('0')


        else:
            # 2. 0 padding
            while(len(sentence_list[i])<sentLength+1):
            #while(len(sentence_list[i])<sentLength):
                sentence_list[i].append('0')

    return sentiment_list,sentence_list,raw_word_list

    

def countSentiment(sentiment_list):
    '''
    Function:
        Count and print # of positive and negative comments
        
    Parameters:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]

    '''
    negative=0
    positive=0
    for s in range(len(sentiment_list)):
        if sentiment_list[s]==-1:
            negative+=1
        if sentiment_list[s]==1:
            positive+=1
    print('For',len(sentiment_list),' comments, it has',negative,'negative comments and', positive,' positive comments') 
    

# (1) Segment and clean labeled comments
labeled_comment_path=os.path.abspath('..//Corpus')
labeled_comment_file='ProsCons.txt' 
sentiment_list,sent_list,_=cleanLabeledComment(labeled_comment_path,labeled_comment_file)

# sentence_list=[segmented comment1,segmented comment2,segmented comment3,...,segmented commentn]
sentence_list=[]
for sent in sent_list:
    temp_list=''
    count=0
    for word in sent:
        temp_list=temp_list+' '+word
        count+=1
    sentence_list.append(temp_list)


v=CountVectorizer()
train_data_features=v.fit_transform(sentence_list)
print(train_data_features.shape)


data_train, data_test, label_train, label_test = train_test_split(
                                        train_data_features,sentiment_list,test_size=0.25,random_state = 3)

                                        
countSentiment(sentiment_list)

 
 
SVM_model=SVC()
SVM_model = SVM_model.fit(data_train,label_train)
print(labeled_comment_file+'SVM Accuracy:',SVM_model.score(data_test,label_test))

                                   



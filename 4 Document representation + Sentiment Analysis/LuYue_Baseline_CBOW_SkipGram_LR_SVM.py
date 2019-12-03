# -*- coding: utf-8 -*-

"""
Baseline CBOW + LR & SVM: 
    Train and test baseline CBoW+LR & CBoW+SVM
    (1) Load Word2Vec Model
    (2) Segment and clean labeled comments
    (3) Represent a document(comment) by averaging all word vectors' value in a comment
    (4) Split train & test data
    (5) Train and test machine learning model (LR and SVM)

@author: Yue, Lu
"""
import joblib
import os
import numpy as np
import re
from  sklearn.svm import SVC #support vector classifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
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
            while(len(sentence_list[i])<sentLength+1):
            #while(len(sentence_list[i])<sentLength):
                sentence_list[i].append('0')

    return sentiment_list,sentence_list,raw_word_list
    
    
def commentToVec(comment,model):
    '''
    Function:
        (1) Get vector values of each word in the current comment
        (2) Get current comment vector by averaging all words' the vector values
        
    Parameters:
        comment: single comment with a list of words [word1,word2,word3,...,wordn]
        model: word2vec model

    Return:
        current comment vector
    '''
    word_vec=np.zeros((1,num_dimension))
    for word in comment:
        if word in model:
            word_vec+=np.array([model[word]])


    return word_vec.mean(axis=0)

    
def loadModel(curr_path,model_filename,model_name):
    '''
    Function:
        Load word2vec model from specific path
    Parameters:
        curr_path: project path
        model_filename: folder name of models
        model_name: file name of word2vec model

    Return:
        loaded word2vec model
    '''
    model = joblib.load(os.path.join(curr_path,model_filename,model_name));
    return model
    
if __name__=='__main__':
    
    # (1) Load Word2Vec Model
    num_dimension = 100
    model_name='CBOW_{}dimension.model'.format(num_dimension)
    #model_name='Skip-Gram_{}dimension.model'.format(num_dimension) 
    curr_path=os.path.abspath('..')
    print(curr_path)
    model_filename='Model'       
    model = loadModel(curr_path,model_filename,model_name)
    model.init_sims(replace=True)

    
    # (2) Segment and clean labeled comments
    labeled_comment_path=os.path.abspath('..//Corpus')
    labeled_comment_file='ProsCons.txt'   
    sentiment_list,sentence_list,raw_word_list=cleanLabeledComment(labeled_comment_path,labeled_comment_file)


    # (3) Get comment vectors
    comment_vec=[]
    for sent in sentence_list:
        # get current comment vector
        comment_vec.append(commentToVec(sent,model))
    comment_vec=np.array(comment_vec)
    train_data_features=comment_vec
    
    data_train, data_test, label_train, label_test = train_test_split(
                                                train_data_features,sentiment_list,test_size=0.25,random_state = 3)

    
    # 1. CBOW+LR           
    LR_model=LogisticRegression()
    LR_model = LR_model.fit(data_train,label_train)
    print(labeled_comment_file+' '+model_name+' LR Accuracy:',LR_model.score(data_test,label_test))
    
        
         
    # 2. CBOW+SVM        
    SVM_model=SVC()
    SVM_model = SVM_model.fit(data_train,label_train)
    print(labeled_comment_file+' '+model_name+' SVM Accuracy:',SVM_model.score(data_test,label_test))

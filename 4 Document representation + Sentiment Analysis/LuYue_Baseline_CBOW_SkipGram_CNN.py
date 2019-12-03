# -*- coding: utf-8 -*-
"""
Baseline CBOW + LSTM & Bi-LSTM & stacked Bi-LSTM: 
    Train and test baseline CBoW+LSTM & CBoW+Bi-LSTM & CBoW+stacked Bi-LSTM
    (1) Load Word2Vec Model
    (2) Load labeled comments and preprocess the comments
    (3) Define parameters of LSTM models
    (4) Construct LSTM models
    (5) Define loss function and optimizer
    (6) Calculate the accuracy
    (7) Define model saver
    (8) Train and save the model 
    (9) Load and test the model
    
@author: Yue, Lu
"""
import joblib
import os
import re
import nltk
from sklearn.model_selection import train_test_split
from tensorflow.contrib import rnn 
import tensorflow as tf
import numpy as np


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
 

    return sentiment_list,sentence_list,raw_word_list
    
def sentToVec(sentence):
    '''
    Function:
        Transform a comment into a comment vector
        
    Parameters:
        sentence: a comment with normalized length

    Return:
        sent_vec: a sentence vector after words are transformed into word vectors 
                    E.g[
                        word1[v1,v2,v3,...,vd]
                        word2[v1,v2,v3,...,vd]
                        ...
                        wordk[v1,v2,v3,...,vd]
                        ]
    '''

    sent_vec=np.zeros((1,numDimensions))
    #flag indicates if it's the 1st word in the sent    
    flag=0
    
    for word in sentence:
        word_vec=np.zeros((1,numDimensions))
        
        # current word can be found in word2vec model
        if word in model:
            if word!=0:               
                word_vec=np.array(model[word])
                # get word vector
                
                # current word is the 1st word in comment [ word1[v1,v2,v3,...,vd] ]
                if flag==0:
                    sent_vec+=word_vec
                    flag=1

                # current word is not the 1st word in comment [ word1[v1,v2,v3,...,vd]  word2[v1,v2,v3,...,vd] ...  wordn[v1,v2,v3,...,vd] ]
                else:
                    sent_vec=np.vstack((sent_vec,word_vec))                 


        # current word can not be found in word2vec model
        else:
            tmp=np.zeros((numDimensions,))
            word_vec=np.array(tmp)
            #   set zero vector
            
            # current word is the 1st word in comment [ word1[0,0,0,...,0] ]
            if flag==0:
                sent_vec+=word_vec
                flag=1
            
            # current word is not the 1st word in comment [ word1[v1,v2,v3,...,vd]  word2[v1,v2,v3,...,vd] ...  wordn[0,0,0,...,0] ]
            else:
                sent_vec=np.vstack((sent_vec,word_vec))
         
    return sent_vec
    
def allInput(sentiment_list,sentence_list,raw_word_list):
    '''
    Function:
        Get all inputs of LSTM model
        (1) Calculate the average sentence length K
        (2) Normalize the length of sentence
        (3) Concat all comment vectors
        (4) Encode all labels using one-hot
        
    Parameters:
        sentiment_list: a list of labels    [label1,label2,label3,...,labeln]
        sentence_list:  a list of comments  [[word1,word2,word3,...,wordn],[word1,word2,word3,...,wordn],...,[word1,word2,word3,...,wordn]]
        raw_word_list:  a list of all words [word1,word2,word3,...,wordm]

    Return:
        all_sentiment_vec:
        [
            sentiment1[0,1]
            sentiment2[0,1]
            ...
        ]
        
        all_sentence_vec:
        [
            sentence1[
                        word1[v1,v2,v3,..,vd]
                        word2[v1,v2,v3,..,vd]
                        ...
                        wordk[v1,v2,v3,..,vd]
                    ]  
                    
            sentence2[
                        word1[v1,v2,v3,..,vd]
                        word2[v1,v2,v3,..,vd]
                        ...
                        wordk[v1,v2,v3,..,vd]
                    ] 
                ...
            
        ]

    '''
    # (1) Calculate the average sentence length K
    sentLength=len(raw_word_list)/len(sentence_list)+5


    # (2) Normalize the length of sentence:
    #       length of comment > average sentence length => extract key words using TF-IDF
    #       length of comment < average sentence length => 0-padding
    for i in range(len(sentence_list)):
        if len(sentence_list[i])>sentLength:
            #sentence_list[i]=jieba.analyse.extract_tags(str(sentence_list[i]),topK=int(sentLength+1))
            sentence_list[i]=sentence_list[i][:int(sentLength+1)]
            #while(len(sentence_list[i])<sentLength+1):
            while(len(sentence_list[i])<sentLength):
                sentence_list[i].append(0)


        else:
            #while(len(sentence_list[i])<sentLength+1):
            while(len(sentence_list[i])<sentLength):
                sentence_list[i].append(0)

                
    
    # (3) Concat all comment vectors
    flag=0
    all_sentence_vec=np.zeros((round(sentLength+1),numDimensions))
    for sent in sentence_list:
        # if current comment is the 1st comment in the list
        if flag==0:
            sent_vec=sentToVec(sent)
            all_sentence_vec+=sent_vec
            flag=1
            
        # if current comment is not the 1st comment in the list
        else:
            sent_vec=sentToVec(sent)
            all_sentence_vec=np.vstack((all_sentence_vec,sent_vec))
    
    all_sentence_vec=all_sentence_vec.reshape(-1,round(sentLength+1),numDimensions)
    #label and input data
    
    
    # (4) Encode all labels using one-hot
    flag=0
    # 2 classes
    all_sentiment_vec=np.zeros((2,))
    for sent in sentiment_list:
        if sent==1:
        # Positive -> [0,1]
            sent_vec=np.array([0,1])
            # if current label is the 1st label in the list
            if flag==0:
                all_sentiment_vec+=sent_vec
                flag=1
            else:
                all_sentiment_vec=np.vstack((all_sentiment_vec,sent_vec))
                
                
        if sent==-1:
        # Negative -> [1,0]
            sent_vec=np.array([1,0])
            # if current label is the 1st label in the list
            if flag==0:
                all_sentiment_vec+=sent_vec
                flag=1
            else:
                all_sentiment_vec=np.vstack((all_sentiment_vec,sent_vec))
    
    #print(all_sentiment_vec.shape)
    return all_sentiment_vec,all_sentence_vec
   

def batchInput(train_data,train_label,batch_size):
    '''
    Function:
        Segment the training set according to the batch size
        
    Parameters:
        train_data: training comment vectors
        train_label: training comment labels
        batch_size: size of network input

    Return:
        batch_data: segmented training data [[batch1],[batch2],[batch3],...]
        batch_label: segmented training label [[batch1],[batch2],[batch3],...]
        batch_num: number of batch for one epoch
    '''

    train_num=train_data.shape[0]
    p=0
    batch_num=0
    batch_data=[]
    batch_label=[]
    
    if(train_num % batch_size!=0):
        batch_num=int(train_num/batch_size)+1
        for i in range(batch_num):
            if i==(batch_num-1):
                batch_size1=train_num-p
                batch_range=p+batch_size1
                tmp_data=train_data[p:batch_range,:,:]
                tmp_label=train_label[p:batch_range,:]
                #print(tmp_data.shape)
                #print(tmp_label.shape)
                batch_data.append(tmp_data)
                batch_label.append(tmp_label)
                p=p+batch_size1
            else:
                batch_range=p+batch_size
                tmp_data=train_data[p:batch_range,:,:]
                tmp_label=train_label[p:batch_range,:]
                #print(tmp_data.shape)
                #print(tmp_label.shape)
                batch_data.append(tmp_data)
                batch_label.append(tmp_label)
                p=p+batch_size
                          
            
    else:
        batch_num=int(train_num/batch_size)
        for i in range(batch_num):
            batch_range=p+batch_size
            tmp_data=train_data[p:batch_range,:,:]
            tmp_label=train_label[p:batch_range,:]
            batch_data.append(tmp_data)
            batch_label.append(tmp_label)
            p=p+batch_size
    
    return batch_data,batch_label,batch_num


def LSTM(lstmUnits,keepratio,data,numClasses):
    '''
    Function:
        Construct a LSTM model
        
    Parameters:
        lstmUnits:  hidden size for each layer
        keepratio:  keep ratio of drop out
        data:       Input data of model
        numClasses: Num of different classes

    Return:
        prediction: predicted value
        weight:     weight of last layer
    '''
    #LSTM
    # hidden cell
    lstmCell=tf.contrib.rnn.BasicLSTMCell(lstmUnits)
    # dropout
    lstmCell=tf.contrib.rnn.DropoutWrapper(cell=lstmCell,output_keep_prob=keepratio)
    # last hidden vector
    value,_=tf.nn.dynamic_rnn(lstmCell,data,dtype=tf.float32)

    weight=tf.Variable(tf.truncated_normal([lstmUnits,numClasses]))
    bias=tf.Variable(tf.constant(0.1,shape=[numClasses]))
    value=tf.transpose(value,[1,0,2])
    #tf.gather(): get the value of the last cell
    last=tf.gather(value,int(value.get_shape()[0])-1)
    prediction=(tf.matmul(last,weight)+bias)
    return prediction,weight

    
def lstm_cell(hidden_size,keep_prob):  
    '''
    Function:
        Construct a hidden LSTM cell
        
    Parameters:
        hidden_size: hidden size for each layer
        keep_prob:   keep ratio of drop out

    Return:
        a hidden LSTM cell
    '''
    cell = rnn.LSTMCell(hidden_size, reuse=tf.get_variable_scope().reuse)  
    return rnn.DropoutWrapper(cell, output_keep_prob=keep_prob)  
        


def CNN(lstmUnits,keepratio,data,kernel_size,numClasses):

    conv = tf.layers.conv1d(data, lstmUnits, kernel_size, name='conv')
    gmp = tf.reduce_max(conv, reduction_indices=[1], name='gmp')
    hidden_dim=128
    fc = tf.layers.dense(gmp, hidden_dim, name='fc1')
    fc = tf.contrib.layers.dropout(fc, keepratio)
    fc = tf.nn.relu(fc)
    prediction = tf.layers.dense(fc, numClasses, name='fc2')
    return prediction

    
if __name__=='__main__':
    kernel_size2=3
    # (1) Load Word2Vec Model
    # Wordvec dimension  
    numDimensions = 100
         
    #model_name='CBOW_{}dimension.model'.format(numDimensions)   
    model_name='Skip-Gram_{}dimension.model'.format(numDimensions)

   
    curr_path=os.path.abspath('..')
    model_filename='Model'
      
    model = loadModel(curr_path,model_filename,model_name)
        # if you don't plan to train the model any further, calling
        # init_sims make the model much more memory-efficient
    model.init_sims(replace=True)
    

    
    # (2) Load labeled comments and preprocess the comments
    labeled_comment_path=os.path.abspath('..//Corpus')
    labeled_comment_file='IMDB1.txt'    
    sentiment_list,sentence_list,raw_word_list=cleanLabeledComment(labeled_comment_path,labeled_comment_file)
    


    tf.reset_default_graph()
    

    # (3) Define parameters of LSTM models
    # Size of network inputs                                                  #
    sentLength      = round(len(raw_word_list)/len(sentence_list)+1)+5        # Sentence Length
    # Num of forward and backward layer of Bi-LSTM250                         #
    num_layers      = 2                                                       # Layer Num
    # Hidden size for each layers                                             #
    lstmUnits       = 64                                                      # LSTM Cell
    # Num of different classes (sentiment_list不重复元素数)
    numClasses      = len(set(sentiment_list))
    # kernel_size
    kernel_size = 5
    #kernel_size = 3
    # Learning rate
    lr = 1e-3
    # Num of epoch
    training_epochs = 1200   
    # Keep ratio of drop out
    keepratio = tf.placeholder(tf.float32)  
    # True label of inputs (batchSize,numClasses)
    labels=tf.placeholder(tf.float32,[None,numClasses])
    # Inputs of network (batchSize,sentLength,numDimensions)
    data=tf.placeholder(tf.float32,[None,sentLength,numDimensions])

    batch_size      = 300
    display_step    = 1
    save_step       = 1500

    # (4) Construct CNN models
    #prediction,W=LSTM(lstmUnits,keepratio,data,numClasses)                    
    prediction=CNN(lstmUnits,keepratio,data,kernel_size,numClasses)


    
    # (5) Define loss function and optimizer

    #CNN_loss
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=prediction,labels=labels))

   ## Adam
    optm = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss) 


    # (6) Calculate the accuracy
    corr = tf.equal(tf.argmax(prediction, 1), tf.argmax(labels, 1)) 
    accr = tf.reduce_mean(tf.cast(corr, "float"))
    init = tf.global_variables_initializer() 
    

    # (7) Define model saver
    saver=tf.train.Saver(max_to_keep=3)
   
   
    sess = tf.Session()
    sess.run(init)
    all_labels,all_data=allInput(sentiment_list,sentence_list,raw_word_list)
    print(all_data.shape)
    print(all_labels.shape)


    
    #do_train=1 train the model
    #do_train=0 test the model
    do_train = 1

    # (8) Train and save the model
    if do_train==1: 
        avg_cost_set=[] 
        epoch_set=[]
        
        train_accr_set=[]
        test_accr_set=[]       
        avg_accr_set=[]
        display_epoch_set=[]  
        
        train_valid_data, test_data, train_valid_label, test_label = train_test_split(
                                    all_data,all_labels,test_size=0.25,random_state=3)                                    
                                    
        for epoch in range(training_epochs):
            
            train_data, valid_data, train_label, valid_label = train_test_split(
                                    train_valid_data,train_valid_label,test_size=0.3)
            total_cost = 0.
            batch_data,batch_label,batch_num=batchInput(train_data,train_label,batch_size)
            
            # Train the model batch by batch
            for i in range(batch_num):
                train_batch_data=batch_data[i]
                train_batch_label=batch_label[i]
                cost = 0.             
                sess.run(optm, feed_dict={labels:train_batch_label,data:train_batch_data,keepratio:0.7})
                # Compute loss
                cost = sess.run(loss, feed_dict={labels:train_batch_label,data:train_batch_data,keepratio:0.7})
                total_cost=total_cost+cost
            avg_cost=total_cost/batch_num
            avg_cost_set.append(avg_cost)
            epoch_set.append(epoch)
    
            # Display logs per epoch step
            if epoch % display_step == 0: 
                #batch_labels,batch_data=allInput(sentiment_list,sentence_list,raw_word_list)
                print ("Epoch: %03d/%03d cost: %.9f" % (epoch, training_epochs, avg_cost))
                train_acc = sess.run(accr, feed_dict={labels:train_label,data:train_data,keepratio:1.0})
  
                print (" Training accuracy: %.3f" % (train_acc))
                #test_acc = sess.run(accr, feed_dict={labels:valid_label,data:valid_data, keepratio:1.0})
                test_acc = sess.run(accr, feed_dict={labels:test_label,data:test_data, keepratio:1.0})
                train_accr_set.append(train_acc)
                test_accr_set.append(test_acc)
                avg_accr_set.append(sum(test_accr_set)/len(test_accr_set))
                display_epoch_set.append(epoch)
                
                print (" Test accuracy: %.3f" % (test_acc))
                print(' Avg test accr:',sum(test_accr_set)/len(test_accr_set))
                print(' Max Accr:',max(test_accr_set), 'for',len(test_accr_set),'epoches')
                print(' ')

                
            #Save the model
            if epoch % save_step ==0 and epoch!=0:
                model_name='Stackedmodel_{}layers'.format(num_layers)
                saver.save(sess,'../Model/'+model_name+str(epoch)+'.ckpt')
        
        print ("OPTIMIZATION FINISHED")
        
        datapath=os.path.abspath('../Data')
        dataname=model_name+'.txt'
        file=open(datapath+'/'+dataname,'w')
        for i in range(len(train_accr_set)):
            file.write(str(train_accr_set[i])+'\t'+str(test_accr_set[i])+'\t'+str(avg_cost_set[i])+'\n')
        file.close()
        

    # (9) Load and test the model
    if do_train==0:
        
 
        saver.restore(sess,tf.train.latest_checkpoint('../Model'))   
        
        # 1. Get test data set
        train_valid_data, test_data, train_valid_label, test_label = train_test_split(
                                    all_data,all_labels,test_size=0.25,random_state=3)
        # 2. Test the loaded model
        test_acc = sess.run(accr, feed_dict={labels:test_label,data:test_data, keepratio:1.})
        print (" Test accuracy: %.3f" % (test_acc))

        

        

        





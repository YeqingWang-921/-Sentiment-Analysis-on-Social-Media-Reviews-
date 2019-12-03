# -*- coding: utf-8 -*-
"""
Visualization hidden size: 
    1. Load data from 'Data' folder
    2. Visualize loss and accuracy with different hidden size


@author: Yue, Lu
"""
import os
import matplotlib.pyplot as plt
import seaborn as sns


#读取数据
path=os.path.abspath('..//Data')

#(1)CBOW
name='CBOW+LSTM+ProsCons.txt'
file=open(path+'/'+name,'r')
line=file.readlines()

cbow_train_acc_set=[]
cbow_test_acc_set=[]
cbow_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    cbow_train_acc_set.append(train_test_loss[0])
    cbow_test_acc_set.append(train_test_loss[1])
    cbow_loss_set.append(train_test_loss[2])

#(1)Skip-Gram
name='Skip-Gram+LSTM+ProsCons.txt'
file=open(path+'/'+name,'r')
line=file.readlines()

skip_train_acc_set=[]
skip_test_acc_set=[]
skip_loss_set=[]

for i in range(1200):
    #print(line[i])
    line[i] = line[i].replace('\n','')
    train_test_loss=line[i].split('\t')
    #print(train_test_loss[2])
    skip_train_acc_set.append(train_test_loss[0])
    skip_test_acc_set.append(train_test_loss[1])
    skip_loss_set.append(train_test_loss[2]) 




#模型可视化
#(1) Loss
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cbow_loss_set,sns.xkcd_rgb['orange'],label='CBoW')
plt.plot(skip_loss_set,sns.xkcd_rgb['blue'],label='Skip-Gram')
#plt.plot(cell128_loss_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Average Loss')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='upper right', fontsize=12,frameon=True,shadow=True)
plt.xlim(0,500)
plt.ylim(0,0.5)
plt.savefig(path+'/Word2vec Average Loss.png')
plt.show()

#(2) Training Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cbow_train_acc_set,sns.xkcd_rgb['orange'],label='CBoW')
plt.plot(skip_train_acc_set,sns.xkcd_rgb['blue'],label='Skip-Gram')
#plt.plot(cell128_train_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Training Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.xlim(0,500)
plt.savefig(path+'/Word2vec Training Accuracy.png')
plt.show()

#(3) Validation Accuracy
sns.set_style('whitegrid')
sns.set_context('paper',font_scale=1.5,rc={"lines.linewidth":2.0})
plt.figure(figsize=(5,3))


plt.plot(cbow_test_acc_set,sns.xkcd_rgb['orange'],label='CBoW')
plt.plot(skip_test_acc_set,sns.xkcd_rgb['blue'],label='Skip-Gram')
#plt.plot(cell128_test_acc_set,sns.xkcd_rgb['blue'],label='128 LSTM cells')
plt.xlabel('Number of Epochs')
plt.ylabel('Test Accuracy')
plt.tight_layout(pad=0.4, w_pad=0.5, h_pad=1.0)
plt.legend(loc='lower right', fontsize=12,frameon=True,shadow=True)
plt.xlim(0,500)
plt.savefig(path+'/Word2vec Test Accuracy.png')
plt.show()


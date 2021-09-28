#!/usr/bin/env python
# coding: utf-8

# ## ![Logo](http://eswcontest.or.kr/images/openImg.jpg) 
# ## [제19회 임베디드 소프트웨어 경진대회] __양치의 정석__ :: Training Program

# ## Library Load
# ***

# In[1]:


import numpy as np
import tensorflow as tf
import tensorflow.keras.layers as tl
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import EarlyStopping
import pandas as pd
from sklearn.model_selection import train_test_split
import os


# ## Init & Parameter
# ***

# In[5]:


seq_length = 60
secs_for_action = 60
is_missing = False
early_stopping = EarlyStopping(patience = 40)


# In[3]:


Hand = ['Left','Right']
Options = ['None','Vertical','Horizontal','LU','RU','LD','RD','LU_in','RU_in','LD_in','RD_in']
L_hands = []
L_mouths = []
R_hands = []
R_mouths = []


# ## Function
# ***

# In[4]:


def Sequence(args):
    full_seq_data = []
    for seq in range(len(args) - seq_length):
        full_seq_data.append(args[seq:seq + seq_length])
    full_seq_data = np.array(full_seq_data)
    return full_seq_data


# ## Training
# ***

# In[6]:


print('::Training Program Start!::')
for hand in Hand:
    if hand == 'Left':
        for option in Options:
            if os.path.isfile('Datasets/Left_%s.npy' % (option)):
                L_hands.append(np.load('Datasets/Left_%s.npy' % (option)))
            else:
                print('{} Dataset are Missing!...'.format(option))
                is_missing = True
                break
        if is_missing:
            break
            
        None_ = np.array(L_hands[0])
        V_ = np.array(L_hands[1])
        H_ = np.array(L_hands[2])
        LU_ = np.array(L_hands[3])
        RU_ = np.array(L_hands[4])
        LD_ = np.array(L_hands[5])
        RD_ = np.array(L_hands[6])
        LU_in_ = np.array(L_hands[7])
        RU_in_ = np.array(L_hands[8])
        LD_in_ = np.array(L_hands[9])
        RD_in_ = np.array(L_hands[10])
        
    else: # hand is Right
        
        for option in Options:
            if os.path.isfile('Datasets/Right_%s.npy' % (option)):
                R_hands.append(np.load('Datasets/Right_%s.npy' % (option)))
            else:
                print('{} Dataset are Missing!...'.format(option))
                is_missing = True
        if is_missing:
            break
            
        None_ = np.array(R_hands[0])
        V_ = np.array(R_hands[1])
        H_ = np.array(R_hands[2])
        LU_ = np.array(R_hands[3])
        RU_ = np.array(R_hands[4])
        LD_ = np.array(R_hands[5])
        RD_ = np.array(R_hands[6])
        LU_in_ = np.array(R_hands[7])
        RU_in_ = np.array(R_hands[8])
        LD_in_ = np.array(R_hands[9])
        RD_in_ = np.array(R_hands[10])
        
    #hand if end
    
    None_ = Sequence(None_)
    V_ = Sequence(V_)
    H_ =Sequence(H_)
    LU_ = Sequence(LU_)
    RU_ = Sequence(RU_)
    LD_ = Sequence(LD_)
    RD_ = Sequence(RD_)
    LU_in_ = Sequence(LU_in_)
    RU_in_ = Sequence(RU_in_)
    LD_in_ = Sequence(LD_in_)
    RD_in_ = Sequence(RD_in_)

    if hand == 'Left':
        X = np.vstack((None_,V_,H_,LU_,RU_,LD_,RD_,LU_in_,RU_in_,LD_in_,RD_in_)).reshape([-1,seq_length,108])
        Y = np.zeros((L_hands[0]).shape[0]-seq_length)
        for i in range(1,len(L_hands)):
            Y = np.hstack((Y,np.ones(np.array(L_hands[i]).shape[0]-seq_length) * i))
        Y = pd.get_dummies(Y)
    else:
        X = np.vstack((None_,V_,H_,LU_,RU_,LD_,RD_,LU_in_,RU_in_,LD_in_,RD_in_)).reshape([-1,seq_length,108])
        Y = np.zeros((R_hands[0]).shape[0]-seq_length)
        for i in range(1,len(R_hands)):
            Y = np.hstack((Y,np.ones(np.array(R_hands[i]).shape[0]-seq_length) * i))
        Y = pd.get_dummies(Y)
    
    print('{} X shape : {} , Y shape : {}'.format(hand,X.shape , Y.shape))
    
    x,x_test,y,y_test = train_test_split(X,Y,test_size=0.2,random_state=8014)
    
    
    model = Sequential()
    model.add(tl.LSTM(64,activation='relu',input_shape=X.shape[1:3]))
    model.add(tl.Dense(32,activation='relu'))
    model.add(tl.Dense(Y.shape[1],activation='softmax'))
    
    model.compile('adam','categorical_crossentropy','accuracy')
    
    model.fit(x,y,validation_data=(x_test,y_test),epochs=100,callbacks=[early_stopping])
    
    os.makedirs('Model/',exist_ok=True)
    model.save('Model/{}_model.h5'.format(hand))
    print('%s Model Saved at Model/%s_model.h5' % (hand,hand))
    
    


# In[ ]:





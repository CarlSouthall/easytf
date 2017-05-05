# -*- coding: utf-8 -*-
"""
Created on Tue May  2 12:39:23 2017

@author: carl
"""

import numpy as np
import tensorflow as tf
from tensorflow.contrib import rnn
import os

class RNN:
    
    def __init__(self,features,targ,val,val_targ,filename,n_hidden=[100,100],n_layers=2,cell_type='LSTMP',dropout=0.25,init_method='zero',truncated=1000,optimizer='Adam',learning_rate=0.003,n_classes=2, minimum_epoch=0,maximum_epoch=1,display_train_loss='True',configuration='R', attention_number=2,display_accuracy='True'):
        self.features=features
        self.targ=targ
        self.val=val
        self.val_targ=val_targ
        self.filename=filename
        self.n_hidden=n_hidden
        self.n_layers=n_layers
        self.cell_type=cell_type
        self.dropout=dropout
        self.configuration=configuration
        self.init_method=init_method
        self.truncated=truncated
        self.optimizer=optimizer
        self.learning_rate=learning_rate
        self.n_classes=n_classes
        self.minimum_epoch=minimum_epoch
        self.maximum_epoch=maximum_epoch
        self.display_train_loss=display_train_loss
        self.n_input = len(self.features[0,0])
        self.num_batch=len(self.features)
        self.val_num_batch=len(self.val)
        self.n_steps = len(self.features[0])
        self.attention_number=attention_number
        self.display_accuracy=display_accuracy 
        
    def cell_create(self):
    	  
        if self.cell_type == 'tanh':
            cells = rnn.MultiRNNCell([rnn.BasicRNNCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
        elif self.cell_type == 'LSTM': 
            cells = rnn.MultiRNNCell([rnn.BasicLSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
        elif self.cell_type == 'GRU':
            cells = rnn.MultiRNNCell([rnn.GRUCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
        elif self.cell_type == 'LSTMP':
            cells = rnn.MultiRNNCell([rnn.LSTMCell(self.n_hidden[i]) for i in range(self.n_layers)], state_is_tuple=True)
        cells = rnn.DropoutWrapper(cells, input_keep_prob=self.dropout_ph,output_keep_prob=self.dropout_ph)         
        return cells
        
    def weight_bias_init(self):
              
        if self.init_method=='zero':
           self.biases = tf.Variable(tf.zeros([self.n_classes]))           
        elif self.init_method=='norm':
    		   self.biases = tf.Variable(tf.random_normal([self.n_classes]))   		   
        if self.configuration =='B':
            if self.init_method=='zero':  
                self.weights = { '1': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number*2)+1)), self.n_classes])),'2': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number*2)+1)), self.n_classes]))}     
            elif self.init_method=='norm':
        		   self.weights = { '1': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number*2)+1)), self.n_classes])),'2': tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number*2)+1)), self.n_classes]))} 
        if self.configuration =='R':
            if self.init_method=='zero':  
                self.weights = tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number)+1)), self.n_classes]))     
            elif self.init_method=='norm':
        		   self.weights = tf.Variable(tf.random_normal([self.n_hidden[(len(self.n_hidden)-1)]*(((self.attention_number)+1)), self.n_classes]))
                
    def create(self):
      
      tf.reset_default_graph()
      self.weight_bias_init()
      self.x_ph = tf.placeholder("float32", [1, self.n_steps, self.n_input])
      self.y_ph = tf.placeholder("float32", [self.n_steps, self.n_classes])
      self.seq=tf.constant(self.truncated,shape=[1]) 
      self.dropout_ph = tf.placeholder("float32")
      self.fw_cell=self.cell_create()
      if self.configuration=='R':
          self.outputs, self.states= tf.nn.dynamic_rnn(self.fw_cell, self.x_ph,
                                            sequence_length=self.seq,dtype=tf.float32)
          self.outputs_zero_padded=tf.pad(self.outputs,[[0,0],[self.attention_number,0],[0,0]])
          self.RNNout1=tf.stack([tf.reshape(self.outputs_zero_padded[:,g:g+(self.attention_number+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number)+1)]) for g in range(self.n_steps)])
          self.presoft=tf.matmul(self.RNNout1, self.weights) + self.biases
      elif self.configuration=='B':
          self.bw_cell=self.cell_create()
          self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.x_ph,
                                        sequence_length=self.seq,dtype=tf.float32)
          self.outputs_zero_padded1=tf.pad(self.outputs[0],[[0,0],[self.attention_number,self.attention_number],[0,0]])
          self.outputs_zero_padded2=tf.pad(self.outputs[1],[[0,0],[self.attention_number,self.attention_number],[0,0]])
          self.RNNout1=tf.stack([tf.reshape(self.outputs_zero_padded1[:,g:g+((self.attention_number*2)+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number*2)+1)]) for g in range(self.n_steps)]) 
          self.RNNout2=tf.stack([tf.reshape(self.outputs_zero_padded2[:,g:g+((self.attention_number*2)+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number*2)+1)]) for g in range(self.n_steps)])
          self.presoft=tf.matmul(self.RNNout1, self.weights['1']) + tf.matmul(self.RNNout2, self.weights['2'])+self.biases
      self.pred=tf.nn.softmax(self.presoft)
      self.cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=self.presoft, labels=self.y_ph))
      if self.optimizer == 'GD':
            self.optimize = tf.train.GradientDescentOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
      elif self.optimizer == 'Adam':
            self.optimize = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost) 
      elif self.optimizer == 'RMS':
            self.optimize = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate).minimize(self.cost) 
      self.correct_pred = tf.equal(tf.argmax(self.pred,1), tf.argmax(self.y_ph,1))
      self.accuracy = tf.reduce_mean(tf.cast(self.correct_pred, tf.float32))
      self.init = tf.global_variables_initializer()
      self.saver = tf.train.Saver()
      
    def train(self):
       
      self.create()
      self.iteration=0
      self.epoch=0
      self.prev_val_loss=100
      self.val_loss=99
      with tf.Session() as sess:
        sess.run(self.init)
        while self.epoch < self.minimum_epoch or self.prev_val_loss > self.val_loss:
            for i in xrange(self.num_batch):
               sess.run(self.optimize, feed_dict={self.x_ph: np.expand_dims(self.features[i],0), self.y_ph: self.targ[i],self.dropout_ph:self.dropout})                                    
               self.iteration+=1
            self.epoch+=1   
            print "Epoch " + str(self.epoch)
            if self.epoch > self.minimum_epoch:
                self.loss_train=[]
                self.loss_val=[]
                if self.display_accuracy=='True':
                    self.acc_train=[]
                    self.acc_val=[]
                for i in xrange(self.val_num_batch):
                    if self.display_accuracy=='True':
                        vl,va=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: np.expand_dims(self.val[i],0), self.y_ph: self.val_targ[i], self.dropout_ph:1})
                        self.loss_val.append(vl)
                        self.acc_val.append(va)
                    else:
                        self.loss_val.append(sess.run(self.cost, feed_dict={self.x_ph: np.expand_dims(self.val[i],0), self.y_ph: self.val_targ[i], self.dropout_ph:1}))
                if  self.display_train_loss=='True': 
                    for i in xrange(self.num_batch): 
                        if self.display_accuracy=='True':
                            tl,ta=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: np.expand_dims(self.features[i],0), self.y_ph: self.targ[i], self.dropout_ph:1})
                            self.loss_train.append(tl)
                            self.acc_train.append(ta)
                        else:
                            self.loss_train.append(sess.run(self.cost, feed_dict={self.x_ph: np.expand_dims(self.features[i],0), self.y_ph: self.targ[i], self.dropout_ph:1}))
                        
                    print "Train Minibatch Loss " + "{:.6f}".format(np.mean(np.array(self.loss_train)))
                    if self.display_accuracy=='True':
                        print "Train Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_train)))
                self.prev_val_loss=self.val_loss
                self.val_loss=np.mean(np.array(self.loss_val))              
                print "Val Minibatch Loss " + "{:.6f}".format(self.val_loss)
                if self.display_accuracy=='True':
                    print "Val Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_val)))
            if self.epoch==self.maximum_epoch:
                break
        print "Optimization Finished!"
        self.saver.save(sess, self.filename)
        self.save_location=os.getcwd()
        
    def implement(self,data):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_location+'/'+self.filename)
            oh=[];
            for i in xrange(len(data)):
                test_len = len(data[i])
                NoBS=int(np.floor(len(data[i])/self.n_steps))
                
                if NoBS == 0:
                   e=np.zeros((1,self.n_steps-test_len,len(data[i][0])))
                   f=np.concatenate((np.expand_dims(data[i],0),e),axis=1)
                   
                   oh.append(sess.run(self.pred, feed_dict={self.x_ph: f,                                           
                                                self.dropout_ph:1}))
                
                   oh[i]=oh[i][:test_len] 
                   
                else:
                   oh.append(sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(data[i][:self.n_steps],0),                            
                                                self.dropout_ph:1}))
                                              
                   for j in xrange(NoBS-1):
                    
                        oh1 = sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(data[i][(j+1)*self.n_steps:(j+2)*self.n_steps]),                                                
                                                    dropoutp:1})
                                                    
                        oh[i]=np.concatenate((oh[i],oh1),axis=0)
                   
                   e=np.zeros((1,self.n_steps*(NoBS+1)-test_len,len(data[i][0])))
                   f=np.concatenate((np.expand_dims(data[i][self.n_steps*(NoBS):len(data[i]-1)],0),e),axis=1)
                   
                   oh1 = sess.run(self.pred, feed_dict={self.x_ph: f,                                        
                                            self.dropout_ph:1})
                                            
                   oh[i]=np.concatenate((oh[i],oh1),axis=0)
                   
                   oh[i]=oh[i][:test_len]
                             
            return oh
            
    def train_output(self,output_layers='True'):
        train_output=[]
        train_layers=[]
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_location+'/'+self.filename)
            for i in xrange(self.num_batch):                	       
                train_output.append(sess.run(self.pred, feed_dict={self.x_ph: self.features[i], self.dropout_ph:1}))
                if output_layers =='True':
                   train_layers.append(sess.run(self.outputs, feed_dict={self.x_ph: self.features[i], self.dropout_ph:1}))
            if output_layers =='True':
                train_output=([train_output,train_layers])
        return train_output
        
    def val_output(self,output_layers='True'):
        val_output=[]
        val_layers=[]
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_location+'/'+self.filename)
            for i in xrange(self.val_num_batch):                	       
                val_output.append(sess.run(self.pred, feed_dict={self.x_ph: self.val[i], self.dropout_ph:1}))
                if val_layers =='True':
                   val_layers.append(sess.run(self.outputs, feed_dict={self.x_ph: self.val[i], self.dropout_ph:1}))
            if val_layers =='True':
                val_output=([val_output,val_layers])
        return val_output
        

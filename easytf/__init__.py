# -*- coding: utf-8 -*-
"""
@author: CarlSouthall

"""


from __future__ import absolute_import, division, print_function

import numpy as np
import tensorflow as tf
import os
from tensorflow.contrib import rnn


class CNN:
    
    def __init__(self,training_data, training_labels, validation_data, validation_labels, mini_batch_locations, network_save_filename, minimum_epoch = 5, maximum_epoch = 10, learning_rate = 0.003, n_classes = 2, optimizer='Adam', conv_filter_shapes=[[5,5,1,5],[5,5,5,10]], conv_strides=[[1,1,1,1],[1,1,1,1]], pool_window_sizes=[[1,1,2,1],[1,1,2,1]], fc_layer_sizes=[100], dropout=0.25, pad='SAME', display_accuracy='True', display_train_loss='True', frames_either_side=[[2,2],[0,0]], input_stride_size=[1,1024]):
        if len(input_stride_size)==2:
            self.dim='2d'
        elif len(input_stride_size)==3:
            self.dim='3d'
        self.frames_either_side=frames_either_side 
        self.input_stride_size=input_stride_size
        self.features=self.zero_pad(training_data)
        self.val=self.zero_pad(validation_data)
        self.targ=training_labels     
        self.val_targ=validation_labels
        self.minibatch_nos=mini_batch_locations
        self.filename=network_save_filename
        self.minimum_epoch=minimum_epoch
        self.maximum_epoch=maximum_epoch
        self.learning_rate=learning_rate
        self.n_classes=n_classes
        self.optimizer=optimizer
        self.conv_filter_shapes=conv_filter_shapes
        self.conv_strides=conv_strides
        self.pool_window_sizes=pool_window_sizes
        self.pool_strides=self.pool_window_sizes
        self.fc_layer_sizes=fc_layer_sizes
        self.dropout=dropout
        self.pad=pad
        self.w_conv=[]
        self.b_conv=[]
        self.h_conv=[]
        self.h_pool=[]
        self.h_drop_batch=[]
        self.batch_size=self.minibatch_nos.shape[1]
        self.num_batch=self.minibatch_nos.shape[0]
        self.conv_layer_out=[]
        self.fc_layer_out=[]
        self.w_fc=[]
        self.b_fc=[]
        self.h_fc=[]
        self.display_accuracy=display_accuracy
        self.display_train_loss=display_train_loss      
        if self.dim=='2d':
            self.batch=np.zeros((self.batch_size,(self.frames_either_side[0][0]+self.frames_either_side[0][1])+self.input_stride_size[0],(self.frames_either_side[1][0]+self.frames_either_side[1][1])+self.input_stride_size[1],1))
            self.batch_targ=np.zeros((self.batch_size,self.targ.shape[2]))
        elif self.dim=='3d':
            self.batch=np.zeros((self.batch_size,(self.frames_either_side[0][0]+self.frames_either_side[0][1])+self.input_stride_size[0],(self.frames_either_side[1][0]+self.frames_either_side[1][1])+self.input_stride_size[1],(self.frames_either_side[2][0]+self.frames_either_side[2][1])+self.input_stride_size[2],1))
            self.batch_targ=np.zeros((self.batch_size,self.targ.shape[3]))
            
        
        
    def conv2d(self,data, weights, conv_strides, pad):
      return tf.nn.conv2d(data, weights, strides=conv_strides, padding=pad)
     
    def max_pool(self,data, max_pool_window, max_strides, pad):
      return tf.nn.max_pool(data, ksize=max_pool_window,
                            strides=max_strides, padding=pad)
    
    def conv3d(self,data, weights, conv_strides, pad):
      return tf.nn.conv3d(data, weights, strides=conv_strides, padding=pad)
     
    def max_pool3d(self,data, max_pool_window, max_strides, pad):
      return tf.nn.max_pool3d(data, ksize=max_pool_window,
                            strides=max_strides, padding=pad)
        
    def weight_init(self,weight_shape):
        weight=tf.Variable(tf.truncated_normal(weight_shape))    
        return weight
        
    def bias_init(self,bias_shape,):   
        bias=tf.Variable(tf.constant(0.1, shape=bias_shape))
        return bias
    
    def batch_dropout(self,data):
        batch_mean, batch_var=tf.nn.moments(data,[0])
        scale = tf.Variable(tf.ones(data.get_shape()))
        beta = tf.Variable(tf.zeros(data.get_shape()))
        h_poolb = tf.nn.batch_normalization(data,batch_mean,batch_var,beta,scale,1e-3)
        return tf.nn.dropout(h_poolb, self.dropout_ph)
        
    def conv_2dlayer(self,layer_num):
        self.w_conv.append(self.weight_init(self.conv_filter_shapes[layer_num]))
        self.b_conv.append(self.bias_init([self.conv_filter_shapes[layer_num][3]]))
        self.h_conv.append(tf.nn.relu(self.conv2d(self.conv_layer_out[layer_num], self.w_conv[layer_num], self.conv_strides[layer_num], self.pad) + self.b_conv[layer_num]))
        self.h_pool.append(self.max_pool(self.h_conv[layer_num],self.pool_window_sizes[layer_num],self.pool_strides[layer_num],self.pad))       
        self.conv_layer_out.append(self.batch_dropout(self.h_pool[layer_num]))
    
    def conv_3dlayer(self,layer_num):
        self.w_conv.append(self.weight_init(self.conv_filter_shapes[layer_num]))
        self.b_conv.append(self.bias_init([self.conv_filter_shapes[layer_num][4]]))
        self.h_conv.append(tf.nn.relu(self.conv3d(self.conv_layer_out[layer_num], self.w_conv[layer_num], self.conv_strides[layer_num], self.pad) + self.b_conv[layer_num]))
        self.conv_layer_out.append(self.max_pool3d(self.h_conv[layer_num],self.pool_window_sizes[layer_num],self.pool_strides[layer_num],self.pad))       
        
    def fc_layer(self,layer_num):
        if layer_num ==0:
            convout=self.conv_layer_out[len(self.conv_layer_out)-1]
            self.fc_layer_out.append(tf.reshape(convout, [self.batch_size,-1]))
            flat_shape=self.fc_layer_out[0].get_shape().as_list()
            self.w_fc.append(self.weight_init([flat_shape[1], self.fc_layer_sizes[layer_num]]))
        else:
            self.w_fc.append(self.weight_init([self.fc_layer_sizes[layer_num-1], self.fc_layer_sizes[layer_num]]))
        self.b_fc.append(self.bias_init([self.fc_layer_sizes[layer_num]]))
        self.h_fc.append(tf.nn.relu(tf.matmul(self.fc_layer_out[layer_num], self.w_fc[layer_num]) + self.b_fc[layer_num]))
        self.fc_layer_out.append(self.batch_dropout(self.h_fc[layer_num]))
        
    def create(self):
         tf.reset_default_graph()
         self.x_ph = tf.placeholder(tf.float32, shape=self.batch.shape)
         self.y_ph = tf.placeholder(tf.float32, shape=self.batch_targ.shape)
         self.dropout_ph = tf.placeholder("float32")
         self.conv_layer_out.append(self.x_ph)
         for i in xrange(len(self.conv_filter_shapes)):
             if self.dim=='2d':
                 self.conv_2dlayer(i)
             elif self.dim=='3d':
                 self.conv_3dlayer(i)
         for i in xrange(len(self.fc_layer_sizes)):
             self.fc_layer(i)
         self.w_out = self.weight_init([self.fc_layer_sizes[len(self.fc_layer_sizes)-1], self.n_classes])
         self.b_out = self.bias_init([self.n_classes])
         self.presoft=tf.matmul(self.fc_layer_out[len(self.fc_layer_out)-1], self.w_out) + self.b_out
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
        
    def zero_pad(self,data):
        if self.dim=='2d':
            out_data=np.zeros((data.shape[0]+self.frames_either_side[0][0]+self.frames_either_side[0][1],data.shape[1]+self.frames_either_side[1][0]+self.frames_either_side[1][1]))
            out_data[self.frames_either_side[0][0]:data.shape[0]+self.frames_either_side[0][0],self.frames_either_side[1][0]:data.shape[1]+self.frames_either_side[1][0]]=data
        elif self.dim=='3d':
            out_data=np.zeros((data.shape[0]+self.frames_either_side[0][0]+self.frames_either_side[0][1],data.shape[1]+self.frames_either_side[1][0]+self.frames_either_side[1][1],data.shape[2]+self.frames_either_side[2][0]+self.frames_either_side[2][1]))
            out_data[self.frames_either_side[0][0]:data.shape[0]+self.frames_either_side[0][0],self.frames_either_side[1][0]:data.shape[1]+self.frames_either_side[1][0],self.frames_either_side[2][0]:data.shape[2]+self.frames_either_side[2][0]]=data
        return out_data
            
    def segment_extract(self,data,targ,seg_location):
        self.seg_location=seg_location
        if self.dim=='2d':
            if len(self.seg_location)==1:
                self.seg_location=np.append(self.seg_location,0)
            out=np.expand_dims(data[(self.seg_location[0]*self.input_stride_size[0]):(self.seg_location[0]*self.input_stride_size[0])+self.frames_either_side[0][0]+self.frames_either_side[0][1]+self.input_stride_size[0],(self.seg_location[1]*self.input_stride_size[1]):(self.seg_location[1]*self.input_stride_size[1])+self.frames_either_side[1][0]+self.frames_either_side[1][1]+self.input_stride_size[1]],2), targ[self.seg_location[0]][self.seg_location[1]]
        elif self.dim=='3d':
            if len(self.seg_location)==1:
                self.seg_location=np.append(self.seg_location,0)
                self.seg_location=np.append(self.seg_location,0)
            elif len(selg.seg_location)==2:
                self.seg_location.append(self.seg_location,0)
            out=np.expand_dims(data[(self.seg_location[0]*self.input_stride_size[0]):(self.seg_location[0]*self.input_stride_size[0])+self.frames_either_side[0][0]+self.frames_either_side[0][1]+self.input_stride_size[0],(self.seg_location[1]*self.input_stride_size[1]):(self.seg_location[1]*self.input_stride_size[1])+self.frames_either_side[1][0]+self.frames_either_side[1][1]+self.input_stride_size[1],(self.seg_location[2]*self.input_stride_size[2]):(self.seg_location[2]*self.input_stride_size[2])+self.frames_either_side[2][0]+self.frames_either_side[2][1]+self.input_stride_size[2]],3), targ[self.seg_location[0]][self.seg_location[1]][self.seg_location[2]]   
        return out

    def segment_extract_test(self,data,seg_location):      
        self.seg_location=seg_location
        if self.dim=='2d':
            if len(self.seg_location)==1:
                self.seg_location=np.append(self.seg_location,0)
            out=np.expand_dims(data[(self.seg_location[0]*self.input_stride_size[0]):(self.seg_location[0]*self.input_stride_size[0])+self.frames_either_side[0][0]+self.frames_either_side[0][1]+self.input_stride_size[0],(self.seg_location[1]*self.input_stride_size[1]):(self.seg_location[1]*self.input_stride_size[1])+self.frames_either_side[1][0]+self.frames_either_side[1][1]+self.input_stride_size[1]],2)
        elif self.dim=='3d':
            if len(selg.seg_location)==1:
                self.seg_location=np.append(self.seg_location,0)
                self.seg_location=np.append(self.seg_location,0)
            elif len(selg.seg_location)==2:
                self.location.append(self.seg_location,0)
            out=np.expand_dims(data[(self.seg_location[0]*self.input_stride_size[0]):(self.seg_location[0]*self.input_stride_size[0])+self.frames_either_side[0][0]+self.frames_either_side[0][1]+self.input_stride_size[0],(self.seg_location[1]*self.input_stride_size[1]):(self.seg_location[1]*self.input_stride_size[1])+self.frames_either_side[1][0]+self.frames_either_side[1][1]+self.input_stride_size[1],(self.seg_location[2]*self.input_stride_size[2]):(self.seg_location[2]*self.input_stride_size[2])+self.frames_either_side[2][0]+self.frames_either_side[2][1]+self.input_stride_size[2]],3)
        return out

    def train(self):
       
      self.create()
      self.iteration=0
      self.epoch=0
      self.prev_val_loss=100
      self.val_loss=99
      self.val_output_dim1=int(self.val.shape[0]/self.input_stride_size[0])
      self.val_output_dim2=int(self.val.shape[1]/self.input_stride_size[1])
      if self.dim=='2d':
          self.val_locations=self.locations_create([self.val_output_dim1,self.val_output_dim2])
      elif self.dim=='3d':
          self.val_output_dim3=self.val.shape[2]/self.input_stride_size[2]
          self.val_locations=self.locations_create([self.val_output_dim1,self.val_output_dim2,self.val_output_dim3])
      with tf.Session() as sess:
        sess.run(self.init)
        while self.epoch < self.minimum_epoch or self.prev_val_loss > self.val_loss:
            for i in xrange(self.num_batch):
               for j in xrange(self.batch_size):
                   self.batch[j],self.batch_targ[j]=self.segment_extract(self.features,self.targ,self.minibatch_nos[i][j])
               sess.run(self.optimize, feed_dict={self.x_ph: self.batch, self.y_ph: self.batch_targ,self.dropout_ph:self.dropout})
               self.iteration+=1
            self.epoch+=1   
            print("Epoch " + str(self.epoch))
            if self.epoch > self.minimum_epoch:
                self.loss_train=[]
                self.loss_val=[]
                if self.display_accuracy=='True':
                    self.acc_train=[]
                    self.acc_val=[]
                self.val_counter=0
                for i in xrange(len(self.val_locations)/self.batch_size):
                    if self.display_accuracy=='True':
                        for j in xrange(self.batch_size):
                            self.batch[j],self.batch_targ[j]=self.segment_extract(self.val,self.val_targ,self.val_locations[self.val_counter])       
                        vl,va=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: self.batch, self.y_ph: self.batch_targ,self.dropout_ph:1})
                        self.loss_val.append(vl)
                        self.acc_val.append(va)
                        self.val_counter+=1        
                    else:
                        for j in xrange(self.batch_size):
                            self.batch[j],self.batch_targ[j]=self.segment_extract(self.val,self.val_targ,self.val_locations[self.val_counter])
                        self.val_counter+=1            
                        self.loss_val.append(sess.run(self.cost, feed_dict={self.x_ph: self.batch, self.y_ph: self.batch_targ,self.dropout_ph:1}))
                if  self.display_train_loss=='True': 
                    for i in xrange(self.num_batch): 
                        if self.display_accuracy=='True':
                            for j in xrange(self.batch_size):
                                self.batch[j],self.batch_targ[j]=self.segment_extract(self.features,self.targ,self.minibatch_nos[i][j])
                            tl,ta=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: self.batch, self.y_ph: self.batch_targ,self.dropout_ph:1})
                            self.loss_train.append(tl)
                            self.acc_train.append(ta)
                        else:
                            for j in xrange(self.batch_size):
                                self.batch[j],self.batch_targ[j]=self.segment_extract(self.features,self.targ,self.minibatch_nos[i][j])
                            self.loss_train.append(sess.run(self.cost, feed_dict={self.x_ph: self.batch, self.y_ph: self.batch_targ,self.dropout_ph:1}))
                        
                    print("Train Minibatch Loss " + "{:.6f}".format(np.mean(np.array(self.loss_train))))
                    if self.display_accuracy=='True':
                        print("Train Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_train))))
                self.prev_val_loss=self.val_loss
                self.val_loss=np.mean(np.array(self.loss_val))              
                print("Val Minibatch Loss " + "{:.6f}".format(self.val_loss))
                if self.display_accuracy=='True':
                    print("Val Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_val))))
            if self.epoch==self.maximum_epoch:
                break
        print("Optimization Finished!")
        self.saver.save(sess, self.filename)
        self.save_location=os.getcwd()
        
    def locations_create(self,sizes):
        locations=[]
        if self.dim=='2d':
            for i in xrange(sizes[0]):
                for j in xrange(sizes[1]):
                    locations.append([i,j])
        elif self.dim=='3d':
            for i in xrange(sizes[0]):
                for j in xrange(sizes[1]):
                    for k in xrange(sizes[2]):
                        locations.append([i,j,k])
        self.dif=len(locations)%self.batch_size
        for i in xrange(self.batch_size-self.dif):
            if self.dim=='2d':
                locations.append([0,0])
            elif self.dim=='3d':
                locations.append([0,0,0])        
        return np.array(locations)
                   
    def implement(self,data):
        with tf.Session() as sess:
            self.saver.restore(sess, self.save_location+'/'+self.filename)
            self.output=[]            
            for i in xrange(len(data)):
                outputdim1=data[i].shape[0]/self.input_stride_size[0]
                outputdim2=data[i].shape[1]/self.input_stride_size[1]
                if self.dim=='2d':
                    self.output.append(np.zeros((outputdim1,outputdim2,self.n_classes)))
                elif self.dim=='3d':
                    outputdim3=data[i].shape[2]/self.input_stride_size[2]
                    self.output.append(np.zeros((outputdim1,outputdim2,outputdim3,self.n_classes)))
                data[i]=self.zero_pad(data[i])
                if self.dim=='2d':
                    self.locations=self.locations_create([outputdim1,outputdim2])
                elif self.dim=='3d':
                    self.locations=self.locations_create([outputdim1,outputdim2,outputdim3])
                self.counter=0
                for j in xrange(len(self.locations)/self.batch_size):
                    for z in xrange(self.batch_size):
                            self.batch[z]=self.segment_extract_test(data[i],self.locations[self.counter])                 
                            self.counter+=1
                    batch_out=sess.run(self.pred, feed_dict={self.x_ph: self.batch, self.dropout_ph:1})
                    
                    if j == len(self.locations)/self.batch_size:
                        implement_length=self.dif
                    else:
                        implement_length=self.batch_size
                    for z in xrange(implement_length):
                        if self.dim=='2d':
                            self.output[i][self.locations[self.counter-self.batch_size+z][0],self.locations[self.counter-self.batch_size+z][1]]=batch_out[z] 
                        elif self.dim=='3d':
                            self.output[i][self.locations[self.counter-self.batch_size+z][0],self.locations[self.counter-self.batch_size+z][1],self.locations[self.counter-self.batch_size+z][2]]=batch_out[z]
            return self.output


class RNN:
    
     def __init__(self, training_data, training_labels, validation_data, validation_labels, mini_batch_locations, network_save_filename, minimum_epoch=5, maximum_epoch=10, n_hidden=[100,100], n_classes=2, cell_type='LSTMP', configuration='B', attention_number=0, dropout=0.25, init_method='zero', truncated=1000, optimizer='Adam', learning_rate=0.003 ,display_train_loss='True', display_accuracy='True'):
         self.features=training_data
         self.targ=training_labels
         self.val=validation_data
         self.val_targ=validation_labels
         self.mini_batch_locations=mini_batch_locations
         self.filename=network_save_filename
         self.n_hidden=n_hidden
         self.n_layers=len(self.n_hidden)
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
         self.num_batch=len(self.mini_batch_locations)
         self.batch_size=self.mini_batch_locations.shape[1]
         self.attention_number=attention_number
         self.display_accuracy=display_accuracy
         self.batch=np.zeros((self.batch_size,self.features.shape[1]))
         self.batch_targ=np.zeros((self.batch_size,self.targ.shape[2]))
        
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
       self.x_ph = tf.placeholder("float32", [1, self.batch.shape[0], self.batch.shape[1]])
       self.y_ph = tf.placeholder("float32", self.batch_targ.shape)
       self.seq=tf.constant(self.truncated,shape=[1]) 
       self.dropout_ph = tf.placeholder("float32")
       self.fw_cell=self.cell_create()
       if self.configuration=='R':
           self.outputs, self.states= tf.nn.dynamic_rnn(self.fw_cell, self.x_ph,
                                             sequence_length=self.seq,dtype=tf.float32)
           self.outputs_zero_padded=tf.pad(self.outputs,[[0,0],[self.attention_number,0],[0,0]])
           self.RNNout1=tf.stack([tf.reshape(self.outputs_zero_padded[:,g:g+(self.attention_number+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number)+1)]) for g in range(self.batch_size)])
           self.presoft=tf.matmul(self.RNNout1, self.weights) + self.biases
       elif self.configuration=='B':
           self.bw_cell=self.cell_create()
           self.outputs, self.states= tf.nn.bidirectional_dynamic_rnn(self.fw_cell, self.bw_cell, self.x_ph,
                                         sequence_length=self.seq,dtype=tf.float32)
           self.outputs_zero_padded1=tf.pad(self.outputs[0],[[0,0],[self.attention_number,self.attention_number],[0,0]])
           self.outputs_zero_padded2=tf.pad(self.outputs[1],[[0,0],[self.attention_number,self.attention_number],[0,0]])
           self.RNNout1=tf.stack([tf.reshape(self.outputs_zero_padded1[:,g:g+((self.attention_number*2)+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number*2)+1)]) for g in range(self.batch_size)]) 
           self.RNNout2=tf.stack([tf.reshape(self.outputs_zero_padded2[:,g:g+((self.attention_number*2)+1)],[self.n_hidden[(len(self.n_hidden)-1)]*((self.attention_number*2)+1)]) for g in range(self.batch_size)])
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
       
     def locations_create(self,size):
         self.locations=range(size)
         self.dif=size%self.batch_size
         if self.dif>0:
             for i in xrange(self.batch_size-self.dif):
                 self.locations=np.append(self.locations,0)
         self.location_new=np.reshape(self.locations,[-1,self.batch_size])     
         return self.location_new
    
      
     def train(self):
       
       self.create()
       self.iteration=0
       self.epoch=0
       self.prev_val_loss=100
       self.val_loss=99
       self.val_locations=self.locations_create(len(self.val))
       with tf.Session() as sess:
         sess.run(self.init)
         while self.epoch < self.minimum_epoch or self.prev_val_loss > self.val_loss:
             for i in xrange(self.num_batch):
                for j in xrange(self.batch_size):
                     self.batch[j]=self.features[self.mini_batch_locations[i,j]]
                     self.batch_targ[j]=self.targ[self.mini_batch_locations[i,j]]
                sess.run(self.optimize, feed_dict={self.x_ph: np.expand_dims(self.batch,0), self.y_ph: self.batch_targ,self.dropout_ph:self.dropout})                                    
                self.iteration+=1
             self.epoch+=1   
             print("Epoch " + str(self.epoch))
             if self.epoch > self.minimum_epoch:
                 self.loss_train=[]
                 self.loss_val=[]
                 if self.display_accuracy=='True':
                     self.acc_train=[]
                     self.acc_val=[]
                 for i in xrange(len(self.val_locations)):
                     for j in xrange(self.batch_size):
                             self.batch[j]=self.features[self.val_locations[i,j]]
                             self.batch_targ[j]=self.targ[self.val_locations[i,j]]
                     if self.display_accuracy=='True':
                         vl,va=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: np.expand_dims(self.batch,0), self.y_ph: self.batch_targ, self.dropout_ph:1})
                         self.loss_val.append(vl)
                         self.acc_val.append(va)
                     else:
                         self.loss_val.append(sess.run(self.cost, feed_dict={self.x_ph: np.expand_dims(self.batch,0), self.y_ph: self.batch_targ, self.dropout_ph:1}))
                 if  self.display_train_loss=='True': 
                     for i in xrange(self.num_batch):
                         for j in xrange(self.batch_size):
                             self.batch[j]=self.features[self.mini_batch_locations[i,j]]
                             self.batch_targ[j]=self.targ[self.mini_batch_locations[i,j]]
                         if self.display_accuracy=='True':
                             tl,ta=sess.run((self.cost,self.accuracy), feed_dict={self.x_ph: np.expand_dims(self.batch,0), self.y_ph: self.batch_targ, self.dropout_ph:1})
                             self.loss_train.append(tl)
                             self.acc_train.append(ta)
                         else:
                             self.loss_train.append(sess.run(self.cost, feed_dict={self.x_ph: np.expand_dims(self.batch,0), self.y_ph: self.batch_targ, self.dropout_ph:1}))
                        
                     print("Train Minibatch Loss " + "{:.6f}".format(np.mean(np.array(self.loss_train))))
                     if self.display_accuracy=='True':
                         print("Train Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_train))))
                 self.prev_val_loss=self.val_loss
                 self.val_loss=np.mean(np.array(self.loss_val))              
                 print("Val Minibatch Loss " + "{:.6f}".format(self.val_loss))
                 if self.display_accuracy=='True':
                     print("Val Minibatch Accuracy " + "{:.6f}".format(np.mean(np.array(self.acc_val))))
             if self.epoch==self.maximum_epoch:
                 break
         print("Optimization Finished!")
         self.saver.save(sess, self.filename)
         self.save_location=os.getcwd()
        
     def implement(self,data):
         with tf.Session() as sess:
             self.saver.restore(sess, self.save_location+'/'+self.filename)
             self.test_out=[];
             for i in xrange(len(data)):
                 self.test_len=len(data[i])
                 self.test_locations=self.locations_create(self.test_len)
                 for k in xrange(len(self.test_locations)):
                    for j in xrange(self.batch_size):
                        self.batch[j]=self.features[self.test_locations[k,j]]
                        self.batch_targ[j]=self.targ[self.test_locations[k,j]]
                    if k == 0:
                        self.test_out.append(sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(self.batch,0),self.dropout_ph:1}))                                    
                    elif k > 0:
                        self.test_out_2=sess.run(self.pred, feed_dict={self.x_ph: np.expand_dims(self.batch,0),self.dropout_ph:1})
                        self.test_out[i]=np.concatenate((self.test_out[i],self.test_out_2),axis=0)
                 self.test_out[i]=self.test_out[i][:self.test_len]
                
            
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
        
        


#from __future__ import print_function
import numpy as np
import tensorflow as tf
import os
import sys

from tensorflow.python.ops import rnn_cell,rnn
from tf_data_utils import extract_seq_data


class tf_seqLSTM(object):

    def add_placeholders(self):

        self.batch_len = tf.placeholder(tf.int32,name="batch_len")

        self.max_time = tf.placeholder(tf.int32,name="max_time")
        dim1=self.config.batch_size*(1+self.internal)
        self.input = tf.placeholder(tf.int32,shape=[None,self.config.maxseqlen],name="input")

        self.labels = tf.placeholder(tf.int32,shape=None
                                     ,name="labels")

        self.dropout = tf.placeholder(tf.float32,name="dropout")

        self.lngths = tf.placeholder(tf.int32,shape=None
                                      ,name="lnghts")


    def __init__(self,config
                ):
        self.emb_dim = config.emb_dim
        self.hidden_dim = config.hidden_dim
        self.num_emb = config.num_emb
        self.output_dim = config.output_dim
        self.config=config
        self.batch_size=config.batch_size
        self.reg=self.config.reg
        self.internal=4  #paramter for sampling sequences coresponding to subtrees 
        assert self.emb_dim > 1 and self.hidden_dim > 1

        self.add_placeholders()

        #self.cell = rnn_cell.LSTMCell(self.hidden_dim)

        emb_input = self.add_embedding()

        #self.add_model_variables()

        output_states = self.compute_states(emb_input)

        logits = self.create_output(output_states)

        self.pred = tf.nn.softmax(logits)

        self.loss,self.total_loss = self.calc_loss(logits)

        self.train_op1,self.train_op2 = self.add_training_op()

    def add_embedding(self):
        #embed=np.load('glove{0}_uniform.npy'.format(self.emb_dim))

        with tf.device('/cpu:0'):
            with tf.variable_scope("Embed"):
                embedding=tf.get_variable('embedding',[self.num_emb,
                                                        self.emb_dim]
                                             ,initializer=
                                             tf.random_uniform_initializer(-0.05,0.05),trainable=True,
                                             regularizer=tf.contrib.layers.l2_regularizer(0.0))
                ix=tf.to_int32(tf.not_equal(self.input,-1))*self.input
                emb = tf.nn.embedding_lookup(embedding,ix)
                emb = emb * tf.to_float(tf.not_equal(tf.expand_dims(self.input,2),-1))
                return emb

    def compute_states(self,emb):

        def unpack_sequence(tensor):
            return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))


        with tf.variable_scope("Composition",initializer=
                               tf.contrib.layers.xavier_initializer(),regularizer=
                               tf.contrib.layers.l2_regularizer(self.reg)):
            cell = rnn_cell.LSTMCell(self.hidden_dim)
            #tf.cond(tf.less(self.dropout
            #if tf.less(self.dropout, tf.constant(1.0)):
            cell = rnn_cell.DropoutWrapper(cell,
                                           output_keep_prob=self.dropout,input_keep_prob=self.dropout)
            #output, state = rnn.dynamic_rnn(cell,emb,sequence_length=self.lngths,dtype=tf.float32)
            outputs,_=rnn.rnn(cell,unpack_sequence(emb),sequence_length=self.lngths,dtype=tf.float32)
            #output = pack_sequence(outputs)

        sum_out=tf.reduce_sum(tf.pack(outputs),[0])
        sent_rep = tf.div(sum_out,tf.expand_dims(tf.to_float(self.lngths),1))
        final_state=sent_rep
        return final_state

    def create_output(self,rnn_out):

        with tf.variable_scope("Projection",regularizer=
                               tf.contrib.layers.l2_regularizer(self.reg)):
            U = tf.get_variable("U",[self.output_dim,self.hidden_dim],
                                initializer=tf.random_uniform_initializer(
                                    -0.05,0.05))
            bu = tf.get_variable("bu",[self.output_dim],initializer=
                                 tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(rnn_out,U,transpose_b=True)+bu

            return logits

    def calc_loss(self,logits):

        l1=tf.nn.sparse_softmax_cross_entropy_with_logits(
            logits,self.labels)
        loss=tf.reduce_sum(l1,[0])
        reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
        regpart=tf.add_n(reg_losses)
        total_loss=loss+0.5*regpart
        return loss,total_loss

    def add_training_op_old(self):

        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(self.total_loss)
        return train_op


    def add_training_op(self):
        loss=self.total_loss
        opt1=tf.train.AdagradOptimizer(self.config.lr)
        opt2=tf.train.AdagradOptimizer(self.config.emb_lr)

        ts=tf.trainable_variables()
        gs=tf.gradients(loss,ts)
        gs_ts=zip(gs,ts)

        gt_emb,gt_nn=[],[]
        for g,t in gs_ts:
            if "embedding" in t.name:
                gt_emb.append((g,t))
            else:
                gt_nn.append((g,t))

        train_op2=opt2.apply_gradients(gt_emb)
        train_op1=opt1.apply_gradients(gt_nn)

        train_op=[train_op1,train_op2]

        return train_op

    def train(self,data,sess,isTree=True):

        from random import shuffle
        shuffle(data)
        losses=[]
        for i in range(0,len(data),self.batch_size):
            batch_size = min(i+self.batch_size,len(data))-i
            batch_data=data[i:i+batch_size]

            seqdata,seqlabels,seqlngths,max_len=extract_seq_data(batch_data
                                                         ,self.internal,self.config.maxseqlen)
            feed={self.input:seqdata,self.labels:seqlabels,
                  self.dropout:self.config.dropout,self.lngths:
                  seqlngths,self.batch_len:len(seqdata),self.max_time:max_len}
            #loss,_=sess.run([self.loss,self.train_op],feed_dict=feed)
            loss,_,_=sess.run([self.loss,self.train_op1,self.train_op2],feed_dict=feed)
            #sess.run(self.train_op,feed_dict=feed)

            losses.append(loss)
            avg_loss=np.mean(losses)
            sstr='avg loss %.2f at example %d of %d\r' % (avg_loss, i, len(data))
            sys.stdout.write(sstr)
            sys.stdout.flush()
            #if i>100: break
        return np.mean(losses)

    def evaluate(self,data,sess):
        num_correct=0
        total_data=0
        for i in range(0,len(data),self.batch_size):
            batch_size = min(i+self.batch_size,len(data))-i
            batch_data=data[i:i+batch_size]

            seqdata,seqlabels,seqlngths,max_len=extract_seq_data(batch_data
                                        ,0,self.config.maxseqlen)
            feed={self.input:seqdata,self.labels:seqlabels,
                  self.dropout:1.0,self.lngths:
                  seqlngths,self.batch_len:len(seqdata),self.max_time:max_len}
            pred=sess.run(self.pred,feed_dict=feed)
            y=np.argmax(pred,axis=1)
            #print y,seqlabels,pred
            #print y,seqlabels,pred
            for i,v in enumerate(y):
                if seqlabels[i]==v:
                    num_correct+=1
                total_data+=1
        acc=float(num_correct)/float(total_data)
        return acc





class tf_seqbiLSTM(tf_seqLSTM):

    def add_training_op(self,loss):

        opt = tf.train.AdagradOptimizer(self.config.lr)
        train_op = opt.minimize(loss)
        return train_op

    def compute_states(self,emb):
        def unpack_sequence(tensor):
            return tf.unpack(tf.transpose(tensor, perm=[1, 0, 2]))



        with tf.variable_scope("Composition",initializer=
                               tf.contrib.layers.xavier_initializer(),regularizer=
                               tf.contrib.layers.l2_regularizer(self.reg)):
            cell_fw = rnn_cell.LSTMCell(self.hidden_dim)
            cell_bw = rnn_cell.LSTMCell(self.hidden_dim)
            #tf.cond(tf.less(self.dropout
            #if tf.less(self.dropout, tf.constant(1.0)):
            cell_fw = rnn_cell.DropoutWrapper(cell_fw,
                                           output_keep_prob=self.dropout,input_keep_prob=self.dropout)
            cell_bw=rnn_cell.DropoutWrapper(cell_bw, output_keep_prob=self.dropout,input_keep_prob=self.dropout)

            #output, state = rnn.dynamic_rnn(cell,emb,sequence_length=self.lngths,dtype=tf.float32)
            outputs,_,_=rnn.bidirectional_rnn(cell_fw,cell_bw,unpack_sequence(emb),sequence_length=self.lngths,dtype=tf.float32)
            #output = pack_sequence(outputs)
        sum_out=tf.reduce_sum(tf.pack(outputs),[0])
        sent_rep = tf.div(sum_out,tf.expand_dims(tf.to_float(self.lngths),1))



        final_state=sent_rep
        return final_state




    def create_output(self,rnn_out):

        with tf.variable_scope("Projection",regularizer=
                               tf.contrib.layers.l2_regularizer(self.reg)):
            U = tf.get_variable("U",[self.output_dim,2*self.hidden_dim],
                                initializer=tf.random_uniform_initializer(
                                    -0.05,0.05))
            bu = tf.get_variable("bu",[self.output_dim],initializer=
                                 tf.constant_initializer(0.0),
                                 regularizer=tf.contrib.layers.l2_regularizer(0.0))

            logits=tf.matmul(rnn_out,U,transpose_b=True)+bu

            return logits



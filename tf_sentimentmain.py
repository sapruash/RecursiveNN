import tf_data_utils as utils

import os
import sys
import numpy as np
import tensorflow as tf
import random
import pickle

import tf_seq_lstm
import tf_tree_lstm

DIR = './project_data/sst/'
GLOVE_DIR ='./'




import pdb
import time

#from tf_data_utils import extract_tree_data,load_sentiment_treebank

class Config(object):

    num_emb=None

    emb_dim = 300
    hidden_dim = 150
    output_dim=None
    degree = 2

    num_epochs = 1
    early_stopping = 2
    dropout = 0.5
    lr = 0.05
    emb_lr = 0.1
    reg=0.0001

    batch_size = 5
    #num_steps = 10
    maxseqlen = None
    maxnodesize = None
    fine_grained=False
    trainable_embeddings=True
    nonroot_labels=True
    #dependency=True not supported

def train(restore=False):

    config=Config()


    data,vocab = utils.load_sentiment_treebank(DIR,config.fine_grained)

    train_set, dev_set, test_set = data['train'], data['dev'], data['test']
    print 'train', len(train_set)
    print 'dev', len(dev_set)
    print 'test', len(test_set)

    num_emb = len(vocab)
    num_labels = 5 if config.fine_grained else 3
    for _, dataset in data.items():
        labels = [label for _, label in dataset]
        assert set(labels) <= set(xrange(num_labels)), set(labels)
    print 'num emb', num_emb
    print 'num labels', num_labels

    config.num_emb=num_emb
    config.output_dim = num_labels

    config.maxseqlen=utils.get_max_len_data(data)
    config.maxnodesize=utils.get_max_node_size(data)

    print config.maxnodesize,config.maxseqlen ," maxsize"
    #return 
    random.seed()
    np.random.seed()


    with tf.Graph().as_default():

        #model = tf_seq_lstm.tf_seqLSTM(config)
        model = tf_tree_lstm.tf_NarytreeLSTM(config)

        init=tf.initialize_all_variables()
        saver = tf.train.Saver()
        best_valid_score=0.0
        best_valid_epoch=0
        dev_score=0.0
        test_score=0.0
        with tf.Session() as sess:

            sess.run(init)
            start_time=time.time()

            if restore:saver.restore(sess,'./ckpt/tree_rnn_weights')
            for epoch in range(config.num_epochs):
                print 'epoch', epoch
                avg_loss=0.0
                avg_loss = train_epoch(model, train_set,sess)
                print 'avg loss', avg_loss

                dev_score=evaluate(model,dev_set,sess)
                print 'dev-scoer', dev_score

                if dev_score > best_valid_score:
                    best_valid_score=dev_score
                    best_valid_epoch=epoch
                    saver.save(sess,'./ckpt/tree_rnn_weights')

                if epoch -best_valid_epoch > config.early_stopping:
                    break

                print "time per epochis {0}".format(
                    time.time()-start_time)
            test_score = evaluate(model,test_set,sess)
            print test_score,'test_score'

def train_epoch(model,data,sess):

    loss=model.train(data,sess)
    return loss

def evaluate(model,data,sess):
    acc=model.evaluate(data,sess)
    return acc

if __name__ == '__main__':
    if len(sys.argv) > 1:
        restore=True
    else:restore=False
    train(restore)


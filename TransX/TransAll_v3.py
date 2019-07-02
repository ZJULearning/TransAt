import numpy as np
import tensorflow as tf
import os
import time

class TransAllModel(object):
    def __init__(self, params):
        self.entity_num = params['entity_num']
        self.relation_num = params['relation_num']
        self.batch_size = int(params['batch_size'])
        self.margin = float(params['margin'])
        self.embed_size = int(params['embed_size'])
        self.normed = eval(params['normed'])
        self.act = params['activation']
        self.opt = params['opt']

        self.lr = float(params['learning_rate'])
        
        self.L1_flag = eval(params['L1_flag']) if params.has_key('L1_flag') else True
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.pos_h = tf.placeholder(tf.int32, shape=None, name='pos_h')
            self.pos_t = tf.placeholder(tf.int32, shape=None, name='pos_t')
            self.neg_h = tf.placeholder(tf.int32, shape=None, name='neg_h')
            self.neg_t = tf.placeholder(tf.int32, shape=None, name='neg_t')
            self.rel = tf.placeholder(tf.int32, shape=None, name='relation')

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            self.w_ent = tf.get_variable(name='weights_entity', shape=[self.entity_num,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel = tf.get_variable(name='weights_relation', shape=[self.relation_num,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_met_h = tf.get_variable(name='weights_metric_head', shape=[2*self.embed_size,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.b_met_h = tf.get_variable(name='biases_metric_head', shape=[self.embed_size], initializer = tf.zeros_initializer())
            self.w_met_t = tf.get_variable(name='weights_metric_tail', shape=[2*self.embed_size,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.b_met_t = tf.get_variable(name='biases_metric_tail', shape=[self.embed_size], initializer = tf.zeros_initializer())
            #self.w_met_2 = tf.get_variable(name='weights_metric_2', shape=[self.embed_size,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            #self.b_met_2 = tf.get_variable(name='biases_metric_2', shape=[self.embed_size], initializer = tf.zeros_initializer())

    def _normed(self, weights):
        norm = tf.norm(weights, axis=1)
        normed_weights = tf.divide(weights, tf.reshape(norm, [-1,1]))
        #embed = tf.where(tf.less(norm, 1.), weights, normed_weights)
        embed = normed_weights
        return embed
    
    def _dist(self, head, tail, rel):
        if self.normed:
            head = self._normed(head)
            tail = self._normed(tail)
            rel = self._normed(rel)

        head = tf.matmul(tf.concat([head, rel], 1), self.w_met_h) + self.b_met_h
        tail = tf.matmul(tf.concat([tail, rel], 1), self.w_met_t) + self.b_met_t

        if self.act == 'tanh':
            head = tf.nn.tanh(head)
            tail = tf.nn.tanh(tail)
        elif self.act == 'sigmoid':
            head = tf.nn.sigmoid(head)
            tail = tf.nn.sigmoid(tail)
        elif self.act == 'lrelu':
            head = tf.where(tf.less(head,0.), 0.01*head, head)
            tail = tf.where(tf.less(tail,0.), 0.01*tail, tail)
        '''
        head = tf.matmul(head, self.w_met_2) + self.b_met_2
        tail = tf.matmul(tail, self.w_met_2) + self.b_met_2

        if self.act == 'tanh':
            head = tf.nn.tanh(head)
            tail = tf.nn.tanh(tail)
        elif self.act == 'sigmoid':
            head = tf.nn.sigmoid(head)
            tail = tf.nn.sigmoid(tail)
        elif self.act == 'lrelu':
            head = tf.where(tf.less(head,0.), 0.01*head, head)
            tail = tf.where(tf.less(tail,0.), 0.01*tail, tail)
        '''
        if self.L1_flag:
            dist = tf.reduce_sum(tf.abs(head - tail), 1)
        else:
            dist = tf.reduce_sum((head - tail)**2, 1)

        return dist

    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)


            self.pos = self._dist(pos_h_e, pos_t_e, rel_e)
            self.neg = self._dist(neg_h_e, neg_t_e, rel_e)
            
            self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + self.margin, 0), name='loss')

    def _create_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        elif self.opt == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        else:
            raise ValueError('Undefined optimization method.')

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        

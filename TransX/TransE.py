import numpy as np
import tensorflow as tf
import os
import time

class TransEModel(object):
    def __init__(self, params):
        self.entity_num = params['entity_num']
        self.relation_num = params['relation_num']
        self.batch_size = int(params['batch_size'])
        self.margin = float(params['margin'])
        self.embed_size = int(params['embed_size'])
        self.normed = eval(params['normed'])
        self.opt = params['opt']
        self.dorc = eval(params['dorc'])

        self.lr = float(params['learning_rate'])
        
        self.l1_flag = eval(params['l1_flag']) if params.has_key('l1_flag') else True
        
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
            if self.dorc:
                self.w_ent_t = tf.get_variable(name='weights_entity_tail', shape=[self.entity_num,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

    def _normed(self, weights):
        norm = tf.norm(weights, axis=1)
        normed_weights = tf.divide(weights, tf.reshape(norm, [-1,1]))
        embed = tf.where(tf.less(norm, 1.), weights, normed_weights)
        return embed
    
    def _dist(self, head, tail, rel):
        if self.normed:
            head = self._normed(head)
            tail = self._normed(tail)
            rel = self._normed(rel)

        if self.l1_flag:
            dist = tf.reduce_sum(tf.abs(head + rel - tail), 1)
        else:
            dist = tf.reduce_sum((head + rel - tail)**2, 1)

        return dist

    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            if self.dorc:
                pos_t_e = tf.nn.embedding_lookup(self.w_ent_t, self.pos_t)
                neg_t_e = tf.nn.embedding_lookup(self.w_ent_t, self.neg_t)
            else:
                pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
                neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)

            #pos_h_e = tf.stop_gradient(pos_h_e)
            #pos_t_e = tf.stop_gradient(pos_t_e)
            #neg_h_e = tf.stop_gradient(neg_h_e)
            #neg_t_e = tf.stop_gradient(neg_t_e)

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
        

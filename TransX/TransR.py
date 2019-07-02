import numpy as np
import tensorflow as tf
import os
import time

class TransRModel(object):
    def __init__(self, params):
        self.entity_num = params['entity_num']
        self.relation_num = params['relation_num']
        self.batch_size = int(params['batch_size'])
        self.margin = float(params['margin'])
        self.embed_size_e = int(params['embed_size_e'])
        self.embed_size_r = int(params['embed_size_r'])

        self.lr = float(params['learning_rate'])
        
        self.L1_flag = eval(params['L1_flag']) if params.has_key('L1_flag') else True
        self.normed = eval(params['normed']) if params.has_key('normed') else False
        self.opt = params['opt'] if params.has_key('opt') else 'sgd'
        
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
            self.w_ent = tf.get_variable(name='weights_entity', shape=[self.entity_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel = tf.get_variable(name='weights_relation', shape=[self.relation_num,self.embed_size_r], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.m_rel = tf.get_variable(name='matrix_relation', shape=[self.relation_num,self.embed_size_r,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

    def _normed(self, weights):
        norm = tf.norm(weights, axis=1)
        normed_weights = tf.divide(weights, tf.reshape(norm, [-1,1]))
        embed = tf.where(tf.less(norm, 1.), weights, normed_weights)
        return embed

    def _normloss(self, weights):
        norm = tf.norm(weights, axis=1)
        norm = tf.where(tf.less(norm, 1.), tf.zeros_like(norm), norm*norm)
        return norm

    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            mat = tf.nn.embedding_lookup(self.m_rel, self.rel)
           
            if self.normed:
                pos_h_e = self._normed(pos_h_e)
                pos_t_e = self._normed(pos_t_e)
                neg_h_e = self._normed(neg_h_e)
                neg_t_e = self._normed(neg_t_e)
                rel_e = self._normed(rel_e)

            pos_h_e = tf.matmul(mat, tf.reshape(pos_h_e, [-1,self.embed_size_e,1]))
            pos_t_e = tf.matmul(mat, tf.reshape(pos_t_e, [-1,self.embed_size_e,1]))
            neg_h_e = tf.matmul(mat, tf.reshape(neg_h_e, [-1,self.embed_size_e,1]))
            neg_t_e = tf.matmul(mat, tf.reshape(neg_t_e, [-1,self.embed_size_e,1]))
            
            pos_h_e = tf.reshape(pos_h_e, [-1,self.embed_size_r])
            pos_t_e = tf.reshape(pos_t_e, [-1,self.embed_size_r])
            neg_h_e = tf.reshape(neg_h_e, [-1,self.embed_size_r])
            neg_t_e = tf.reshape(neg_t_e, [-1,self.embed_size_r])

            if self.L1_flag:
                pos = tf.reduce_sum(tf.abs(pos_h_e + rel_e - pos_t_e), 1)
                neg = tf.reduce_sum(tf.abs(neg_h_e + rel_e - neg_t_e), 1)
            else:
                pos = tf.reduce_sum((pos_h_e + rel_e - pos_t_e)**2, 1)
                neg = tf.reduce_sum((neg_h_e + rel_e - neg_t_e)**2, 1)
            
            self.pos_h_n = self._normloss(pos_h_e)
            self.pos_t_n = self._normloss(pos_t_e)
            self.neg_h_n = self._normloss(neg_h_e)
            self.neg_t_n = self._normloss(neg_t_e)
            self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0), name='loss')

    def _create_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        elif self.opt == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        elif self.opt == 'pgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
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
        

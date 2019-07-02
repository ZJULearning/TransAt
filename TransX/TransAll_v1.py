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
        self.embed_size_e = int(params['embed_size_e'])
        self.embed_size_r = int(params['embed_size_r'])
        self.act = params['activation']
        self.alpha = float(params['alpha'])

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
            self.a_rel_h_tmp = tf.placeholder(tf.float32, shape=[self.relation_num, self.embed_size_e], name='a_rel_h_tmp')
            self.a_rel_r_tmp = tf.placeholder(tf.float32, shape=[self.relation_num, self.embed_size_e], name='a_rel_r_tmp')
            self.a_rel_t_tmp = tf.placeholder(tf.float32, shape=[self.relation_num, self.embed_size_e], name='a_rel_t_tmp')
            self.cls = tf.placeholder(tf.float32, shape=None, name='cls')

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            self.w_ent = tf.get_variable(name='weights_entity', shape=[self.entity_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel = tf.get_variable(name='weights_relation', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel_h = tf.get_variable(name='weights_relation_head', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.a_rel_h = tf.get_variable(name='attention_relation_head', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel_t = tf.get_variable(name='weights_relation_tail', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.a_rel_t = tf.get_variable(name='attention_relation_tail', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.a_rel_r = tf.get_variable(name='attention_relation_rel', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

    def _reset_at(self):
        with tf.name_scope('reset'):
            self.reset_at = tf.group(self.a_rel_h.assign(self.a_rel_h_tmp), self.a_rel_t.assign(self.a_rel_t_tmp), self.a_rel_r.assign(self.a_rel_r_tmp), name='reset_op')
    
    def _normed(self, weights):
        norm = tf.norm(weights, axis=1)
        normed_weights = tf.divide(weights, tf.reshape(norm, [-1,1]))
        embed = tf.where(tf.less(norm, 1.), weights, normed_weights)
        return embed

    def _dist(self, head, tail, rel, rel_h, rel_t, a_rel_h, a_rel_r, a_rel_t):
        if self.normed:
            head = self._normed(head)
            tail = self._normed(tail)
            rel = self._normed(rel)

        if self.act == 'tanh':
            rel_h = tf.nn.tanh(rel_h)
            rel_t = tf.nn.tanh(rel_t)
        elif self.act == 'sigmoid':
            rel_h = tf.nn.sigmoid(rel_h)
            rel_t = tf.nn.sigmoid(rel_t)
        elif self.act == 'relu':
            rel_h = tf.nn.relu(rel_h)
            rel_t = tf.nn.relu(rel_t)
        elif self.act == 'id':
            pass
        else:
            ValueError('Undefined activation function!')            

        head_1 = a_rel_h  * rel_h * head
        tail_1 = a_rel_t * rel_t * tail
        rel_1 = a_rel_r * rel

        if self.L1_flag:
            dist = tf.reduce_sum(tf.abs(head_1 + rel_1 - tail_1), 1)
        else:
            dist = tf.reduce_sum((head_1 + rel_1 - tail_1)**2, 1)
        
        return dist
    
    def _dist_2(self, pos_h, pos_t, neg_h, neg_t):
        #if self.L1_flag:
        if False:
            d1 = tf.reduce_sum(tf.abs(pos_h - neg_h), 1)
            d2 = tf.reduce_sum(tf.abs(pos_t - neg_t), 1)
        else:
            d1 = tf.reduce_sum((pos_h - neg_h)**2, 1)
            d2 = tf.reduce_sum((pos_t - neg_t)**2, 1)
        
        dist = tf.where(tf.less(self.cls, 2.), d1, d2)

        return dist
    
    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            rel_h_e = tf.nn.embedding_lookup(self.w_rel_h, self.rel)
            rel_t_e = tf.nn.embedding_lookup(self.w_rel_t, self.rel)
            a_rel_h = tf.nn.embedding_lookup(self.a_rel_h, self.rel)
            a_rel_r = tf.nn.embedding_lookup(self.a_rel_r, self.rel)
            a_rel_t = tf.nn.embedding_lookup(self.a_rel_t, self.rel)
           
            #pos_h_e = tf.stop_gradient(pos_h_e)
            #pos_t_e = tf.stop_gradient(pos_t_e)
            #neg_h_e = tf.stop_gradient(neg_h_e)
            #neg_t_e = tf.stop_gradient(neg_t_e)
            #rel_h_e = tf.stop_gradient(rel_h_e)
            #rel_t_e = tf.stop_gradient(rel_t_e)
            a_rel_h = tf.stop_gradient(a_rel_h)
            a_rel_r = tf.stop_gradient(a_rel_r)
            a_rel_t = tf.stop_gradient(a_rel_t)
            
            self.pos = self._dist(pos_h_e, pos_t_e, rel_e, rel_h_e, rel_t_e, a_rel_h, a_rel_r, a_rel_t)
            self.neg = self._dist(neg_h_e, neg_t_e, rel_e, rel_h_e, rel_t_e, a_rel_h, a_rel_r, a_rel_t)
            dd = self._dist_2(pos_h_e, pos_t_e, neg_h_e, neg_t_e)
            pos = tf.where(tf.less(self.cls, 1.), self.pos, tf.zeros_like(dd))
            neg = tf.where(tf.less(self.cls, 1.), self.neg, dd)

            loss = tf.maximum(pos - neg + self.margin, 0)
            self.loss = tf.reduce_sum(tf.where(tf.less(self.cls, 1.), loss, self.alpha*loss), name='loss')
            #self.loss = tf.reduce_sum(tf.maximum(pos - neg + self.margin, 0), name='loss')

    def _create_optimizer(self):
        if self.opt == 'sgd':
            self.optimizer = tf.train.GradientDescentOptimizer(self.lr).minimize(self.loss)
        elif self.opt == 'adam':
            self.optimizer = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
        elif self.opt == 'ftrl':
            #self.optimizer = tf.train.FtrlOptimizer(self.lr, l1_regularization_strength = 0.5, l2_regularization_strength = 0.5).minimize(self.loss)
            self.optimizer = tf.train.FtrlOptimizer(self.lr,l1_regularization_strength = 1, l2_regularization_strength = 1).minimize(self.loss)
        else:
            raise ValueError('Undefined optimization method.')

    def _create_summaries(self):
        with tf.name_scope('summaries'):
            tf.summary.scalar('loss', self.loss)

            self.summary_op = tf.summary.merge_all()

    def build_graph(self):
        self._create_placeholders()
        self._create_embedding()
        self._reset_at()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        

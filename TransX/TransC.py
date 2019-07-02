import numpy as np
import tensorflow as tf
import os
import time

class TransCModel(object):
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
            self.rel_c_h_tmp = tf.placeholder(tf.float32, shape=[self.relation_num,self.embed_size_e], name='rel_center_head_tmp')
            self.rel_c_t_tmp = tf.placeholder(tf.float32, shape=[self.relation_num,self.embed_size_e], name='rel_center_tail_tmp')

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            self.w_ent = tf.get_variable(name='weights_entity', shape=[self.entity_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel = tf.get_variable(name='weights_relation', shape=[self.relation_num,self.embed_size_r], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_c_h = tf.get_variable(name='relation_center_head', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.rel_c_t = tf.get_variable(name='relation_center_tail', shape=[self.relation_num,self.embed_size_e], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            #self.s_rel = tf.get_variable(name='scale_relation', shape=[self.embed_size_e,self.embed_size_r], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.s_rel_h = tf.get_variable(name='scale_relation_head', shape=[self.relation_num,self.embed_size_e,self.embed_size_r], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.s_rel_t = tf.get_variable(name='scale_relation_tail', shape=[self.relation_num,self.embed_size_e,self.embed_size_r], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
    
    def _reset_center(self):
        with tf.name_scope('reset'):
            self.reset_center = tf.group(self.rel_c_h.assign(self.rel_c_h_tmp), self.rel_c_t.assign(self.rel_c_t_tmp), name='reset_op')

    def _normed(self, weights):
        norm = tf.norm(weights, axis=1)
        normed_weights = tf.divide(weights, tf.reshape(norm, [-1,1]))
        embed = tf.where(tf.less(norm, 1.), weights, normed_weights)
        #embed = normed_weights
        return embed

    def _dist(self, head, tail, rel, rel_c_h, rel_c_t, rel_mat_h, rel_mat_t):
        if self.normed:
            head = self._normed(head)
            tail = self._normed(tail)
            rel = self._normed(rel)

        head = head - rel_c_h
        tail = tail - rel_c_t
        #head = tf.matmul(head - rel_c_h, self.s_rel)
        #tail = tf.matmul(tail - rel_c_t, self.s_rel)
        head = tf.matmul(tf.reshape(head, [-1,1,self.embed_size_e]), rel_mat_h)
        tail = tf.matmul(tf.reshape(tail, [-1,1,self.embed_size_e]), rel_mat_t)
        head = tf.reshape(head, [-1,self.embed_size_r])
        tail = tf.reshape(tail, [-1,self.embed_size_r])
        
        if self.L1_flag:
            dist = tf.reduce_sum(tf.abs(head + rel - tail), 1)
        else:
            dist = tf.reduce_sum((head + rel - tail)**2, 1)
        
        return dist
    
    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            rel_c_h = tf.nn.embedding_lookup(self.rel_c_h, self.rel)
            rel_c_t = tf.nn.embedding_lookup(self.rel_c_t, self.rel)
            rel_mat_h = tf.nn.embedding_lookup(self.s_rel_h, self.rel)
            rel_mat_t = tf.nn.embedding_lookup(self.s_rel_t, self.rel)
           
            rel_c_h = tf.stop_gradient(rel_c_h)
            rel_c_t = tf.stop_gradient(rel_c_t)
            #pos_h_e = tf.stop_gradient(pos_h_e)
            #pos_t_e = tf.stop_gradient(pos_t_e)
            #neg_h_e = tf.stop_gradient(neg_h_e)
            #neg_t_e = tf.stop_gradient(neg_t_e)
            #rel_e = tf.stop_gradient(rel_e)
            
            self.pos = self._dist(pos_h_e, pos_t_e, rel_e, rel_c_h, rel_c_t, rel_mat_h, rel_mat_t)
            self.neg = self._dist(neg_h_e, neg_t_e, rel_e, rel_c_h, rel_c_t, rel_mat_h, rel_mat_t)
            
            self.loss = tf.reduce_sum(tf.maximum(self.pos - self.neg + self.margin, 0), name='loss')

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
        self._reset_center()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        

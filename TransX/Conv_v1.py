import numpy as np
import tensorflow as tf
import os
import time

class ConvModel(object):
    def __init__(self, params):
        self.entity_num = params['entity_num']
        self.relation_num = params['relation_num']
        self.batch_size = int(params['batch_size'])
        self.embed_size = int(params['embed_size'])
        self.normed = eval(params['normed'])
        self.act = params['activation']
        self.channel = int(params['channel'])
        self.opt = params['opt']

        self.lr = float(params['learning_rate'])

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
            self.kernel = tf.get_variable('kernel', [3, 3, 1, self.channel], initializer=tf.truncated_normal_initializer())
            self.biases = tf.get_variable('biases', [self.channel], initializer=tf.zeros_initializer())
            self.w_met = tf.get_variable(name='weights_metric', shape=[self.channel*(self.embed_size-2),2], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.b_met = tf.get_variable(name='biases_metric', shape=[2], initializer = tf.zeros_initializer())

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
        
        head = tf.reshape(head, [-1,self.embed_size,1,1])
        tail = tf.reshape(tail, [-1,self.embed_size,1,1])
        rel = tf.reshape(rel, [-1,self.embed_size,1,1])
        tripmat = tf.concat([head, rel, tail], 2)

        conv = tf.nn.conv2d(tripmat, self.kernel, strides=[1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.relu(conv + self.biases)

        conv1 = tf.reshape(conv1, [-1,self.channel*(self.embed_size-2)])
        logits = tf.matmul(conv1, self.w_met) + self.b_met

        dist, sim = tf.split(tf.nn.softmax(logits), [1,1], 1)

        return dist

    def _logits(self, head, tail, rel):
        if self.normed:
            head = self._normed(head)
            tail = self._normed(tail)
            rel = self._normed(rel)

        head = tf.reshape(head, [-1,self.embed_size,1,1])
        tail = tf.reshape(tail, [-1,self.embed_size,1,1])
        rel = tf.reshape(rel, [-1,self.embed_size,1,1])
        tripmat = tf.concat([head, rel, tail], 2)

        conv = tf.nn.conv2d(tripmat, self.kernel, strides=[1, 1, 1, 1], padding='VALID')
        conv1 = tf.nn.relu(conv + self.biases)

        conv1 = tf.reshape(conv1, [-1,self.channel*(self.embed_size-2)])
        logits = tf.matmul(conv1, self.w_met) + self.b_met

        return logits

    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            neg_h_e = tf.nn.embedding_lookup(self.w_ent, self.neg_h)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            neg_t_e = tf.nn.embedding_lookup(self.w_ent, self.neg_t)

            self.pos = self._dist(pos_h_e, pos_t_e, rel_e)

            logits_pos = self._logits(pos_h_e, pos_t_e, rel_e)
            logits_neg = self._logits(neg_h_e, neg_t_e, rel_e)
            logits = tf.concat([logits_pos, logits_neg], 0)

            labels_pos = tf.ones_like(self.pos_h)
            labels_neg = tf.zeros_like(self.neg_h)
            labels = tf.one_hot(tf.concat([labels_pos, labels_neg], 0), 2)
            self.loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=logits), name='loss')

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
        

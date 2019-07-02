import tensorflow as tf
import os
import time

class SkipModel(object):
    def __init__(self, params):
        self.entity_num = params['entity_num']
        self.relation_num = params['relation_num']
        self.batch_size = int(params['batch_size'])
        self.embed_size = int(params['embed_size'])
        self.normed = eval(params['normed'])
        self.opt = params['opt']

        self.lr = float(params['learning_rate'])
        
        self.L1_flag = eval(params['L1_flag']) if params.has_key('L1_flag') else True
        
        self.global_step = tf.Variable(0, dtype=tf.int32, trainable=False, name='global_step')
        
    def _create_placeholders(self):
        with tf.name_scope("data"):
            self.pos_h = tf.placeholder(tf.int32, shape=None, name='pos_h')
            self.pos_t = tf.placeholder(tf.int32, shape=None, name='pos_t')
            self.rel = tf.placeholder(tf.int32, shape=None, name='relation')

    def _create_embedding(self):
        with tf.name_scope('embedding'):
            self.w_ent = tf.get_variable(name='weights_entity', shape=[self.entity_num,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            self.w_rel = tf.get_variable(name='weights_relation', shape=[self.relation_num,self.embed_size], initializer = tf.contrib.layers.xavier_initializer(uniform = False))

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

        if self.L1_flag:
            dist = tf.reduce_sum(tf.abs(head + rel - tail), 1)
        else:
            dist = tf.reduce_sum((head + rel - tail)**2, 1)

        return dist

    def _create_logits_weight(self):
        with tf.name_scope('softmax'):
            self.w = tf.get_variable(name='weights', shape=[self.embed_size,self.entity_num], initializer = tf.contrib.layers.xavier_initializer(uniform = False))
            w = tf.transpose(self._normed(tf.transpose(self.w)))
            self.s = tf.get_variable(name='scale', shape=[1], initializer = tf.constant_initializer(1.))

    def _create_loss(self):
        with tf.name_scope('loss'):
            pos_h_e = tf.nn.embedding_lookup(self.w_ent, self.pos_h)
            pos_t_e = tf.nn.embedding_lookup(self.w_ent, self.pos_t)
            rel_e = tf.nn.embedding_lookup(self.w_rel, self.rel)

            pos_h_e = self._normed(pos_h_e)
            pos_t_e = self._normed(pos_t_e)
            rel_e = self._normed(rel_e)
            #w = tf.transpose(self._normed(tf.transpose(self.w)))
            w = tf.transpose(self._normed(self.w_ent))

            prd_h = pos_t_e - rel_e
            prd_t = pos_h_e + rel_e
            
    

            self.pos = self._dist(pos_h_e, pos_t_e, rel_e)
            
            #losstype = 'softmax'
            #losstype = 'contrastive'
            losstype = 'trip'
            if losstype == 'softmax':
                logits_h = self.s * tf.matmul(prd_h,w)
                logits_t = self.s * tf.matmul(prd_t,w)
                entropy_h = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.pos_h,self.entity_num), logits=logits_h)
                entropy_t = tf.nn.softmax_cross_entropy_with_logits(labels=tf.one_hot(self.pos_t,self.entity_num), logits=logits_t)
                self.loss = tf.reduce_mean(entropy_h + entropy_t, name='loss')
            elif losstype == 'contrastive':
                ww = tf.reduce_sum(tf.square(tf.transpose(w)), 1)
                hh = tf.reduce_sum(tf.square(prd_h), 1)
                logits_h = tf.matmul(prd_h,w)
                D_h = tf.tile(tf.reshape(hh,[-1,1]),[1,self.entity_num]) + tf.reshape(ww, [1,-1]) - 2 * logits_h
                mask_h = tf.one_hot(self.pos_h,self.entity_num)
                loss_h = tf.reduce_sum(mask_h * D_h + (1-mask_h) * tf.maximum(0., 1-D_h),1)
                tt = tf.reduce_sum(tf.square(prd_t), 1)
                logits_t = tf.matmul(prd_t,w)
                D_t = tf.tile(tf.reshape(tt,[-1,1]),[1,self.entity_num]) + tf.reshape(ww, [1,-1]) - 2 * logits_t
                mask_t = tf.one_hot(self.pos_t,self.entity_num)
                loss_t = tf.reduce_sum(mask_t * D_t + (1-mask_t) * tf.maximum(0., 1-D_t),1)
                self.loss = tf.reduce_mean(loss_h + loss_t)
            elif losstype == 'trip':
                ww = tf.reduce_sum(tf.square(tf.transpose(w)), 1)
                hh = tf.reduce_sum(tf.square(prd_h), 1)
                logits_h = tf.matmul(prd_h,w)
                D_h = tf.tile(tf.reshape(hh,[-1,1]),[1,self.entity_num]) + tf.reshape(ww, [1,-1]) - 2 * logits_h
                mask_h = tf.one_hot(self.pos_h,self.entity_num)
                loss_h = tf.reduce_sum(tf.maximum(0., 2. + tf.reshape(tf.reduce_sum(mask_h * D_h, 1),[-1,1]) - D_h),1)
                tt = tf.reduce_sum(tf.square(prd_t), 1)
                logits_t = tf.matmul(prd_t,w)
                D_t = tf.tile(tf.reshape(tt,[-1,1]),[1,self.entity_num]) + tf.reshape(ww, [1,-1]) - 2 * logits_t
                mask_t = tf.one_hot(self.pos_t,self.entity_num)
                loss_t = tf.reduce_sum(tf.maximum(0., 2. + tf.reshape(tf.reduce_sum(mask_t * D_t, 1),[-1,1]) - D_t),1)
                self.loss = tf.reduce_mean(loss_h + loss_t)

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
        self._create_logits_weight()
        self._create_loss()
        self._create_optimizer()
        self._create_summaries()
        

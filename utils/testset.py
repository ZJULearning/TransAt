import numpy as np
import tensorflow as tf
import os
import time
import random
import cPickle as pkl

class TestSet(object):
    # init testset by its dirctory path
    def __init__(self, dir_name, phase):
        # load entity_id
        self.entity2id = {}
        self.id2entity = {}
        with open(os.path.join(dir_name, 'entity2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.entity2id[line_arr[0]] = int(line_arr[1])
                self.id2entity[int(line_arr[1])] = line_arr[0]
        # load relation_id
        self.relation2id = {}
        self.id2relation = {}
        with open(os.path.join(dir_name, 'relation2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.relation2id[line_arr[0]] = int(line_arr[1])
                self.id2relation[int(line_arr[1])] = line_arr[0]
        
        self.entity_size = len(self.entity2id)
        self.relation_size = len(self.relation2id)
        print 'dataset {} has {} entities.'.format(dir_name.split('/')[-1], self.entity_size)
        print 'dataset {} has {} relations.'.format(dir_name.split('/')[-1], self.relation_size)
        # load trip
        self.fb_h = []
        self.fb_r = []
        self.fb_l = []
        self.ok = set()
        if phase == 'val':
            self._load(os.path.join(dir_name, 'valid.txt'), True)
            self._load(os.path.join(dir_name, 'train.txt'), False)
        elif phase == 'test':
            self._load(os.path.join(dir_name, 'test.txt'), True)
            self._load(os.path.join(dir_name, 'train.txt'), False)
            self._load(os.path.join(dir_name, 'valid.txt'), False)
        elif phase == 'train':
            self._load(os.path.join(dir_name, 'train.txt'), True)
        else:
            raise ValueError('Undefined phase for testset.')
    
    # load function: load trip from file
    def _load(self, filename, flag):
        with open(filename) as fr:
            for line in fr:
                line_arr = line.strip().split()
                tmp_head = self.entity2id[line_arr[0]]
                tmp_last = self.entity2id[line_arr[1]]
                tmp_relation = self.relation2id[line_arr[2]]
                if flag:
                    self.fb_h.append(tmp_head)
                    self.fb_l.append(tmp_last)
                    self.fb_r.append(tmp_relation)
                self.ok.add((tmp_head, tmp_relation, tmp_last))
def test_model_link(models, testset, params):
    try:
        len(models)
    except:
        models = [models]
    gpu_num = len(models)
    params['batch_size'] = 2048*gpu_num
    if gpu_num == 4:
        sp = [params['batch_size']/4,params['batch_size']/2,params['batch_size']/4*3]
    gpu_options = tf.GPUOptions(allow_growth=True)
    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, gpu_options=gpu_options)) as sess:
        sess.run(tf.global_variables_initializer())
        for epoch in xrange(int(params['start']),int(params['end'])+1):
            init_var = []
            for item in tf.global_variables():
                if 'global_step' in item.name: continue
                init_var.append(item)
            saver = tf.train.Saver(init_var)
            saver.restore(sess,params['save_fld']+'/model-{}'.format(epoch*int(params['interval'])))
            model = models[0]
            entity, relation = sess.run([model.w_ent, model.w_rel])
            candi_set_head, candi_set_tail = pkl.load(open(params['save_fld']+'/ht_candi_{}.pkl'.format(epoch*int(params['interval']))))
            lsum = 0; lsum_filter = 0
            rsum = 0; rsum_filter = 0
            lp_n = 0
            rp_n = 0
            lsum_r = np.zeros(testset.relation_size)
            lsum_filter_r = np.zeros(testset.relation_size)
            rsum_r = np.zeros(testset.relation_size)
            rsum_filter_r = np.zeros(testset.relation_size)
            lp_n_r = np.zeros(testset.relation_size)
            rp_n_r = np.zeros(testset.relation_size)
            rel_num = np.zeros(testset.relation_size)

            cnt = 0
            for h,l,rel in zip(testset.fb_h, testset.fb_l, testset.fb_r):
                #if cnt % 100 == 0: print cnt
                cnt += 1
                rel_num[rel] += 1
                nbatches = (testset.entity_size-1) / params['batch_size']
                r_a = rel * np.ones(params['batch_size']/gpu_num)
                t_a = l * np.ones(params['batch_size']/gpu_num)
                a = np.zeros(1)
                #ts = time.time()
                for i in xrange(nbatches+1):
                    if i != nbatches:
                        h_a = np.arange(i*params['batch_size'],(i+1)*params['batch_size'])
                    else:
                        h_a = range(i*params['batch_size'],testset.entity_size)
                        h_a.extend(range((i+1)*params['batch_size']-testset.entity_size))
                        h_a = np.array(h_a)
                    if gpu_num == 1:
                        feed_dict = {models[0].pos_h: h_a, models[0].pos_t: t_a, models[0].rel: r_a}
                        tmp_a0 = sess.run(models[0].pos,feed_dict = feed_dict)
                        a = np.concatenate([a,tmp_a0.reshape(-1)])                        
                    elif gpu_num == 4:
                        feed_dict = {models[0].pos_h: h_a[:sp[0]], models[0].pos_t: t_a, models[0].rel: r_a, models[1].pos_h: h_a[sp[0]:sp[1]], models[1].pos_t: t_a, models[1].rel: r_a, models[2].pos_h: h_a[sp[1]:sp[2]], models[2].pos_t: t_a, models[2].rel: r_a, models[3].pos_h: h_a[sp[2]:], models[3].pos_t: t_a, models[3].rel: r_a}
                        tmp_a0, tmp_a1, tmp_a2, tmp_a3 = sess.run([models[0].pos, models[1].pos, models[2].pos, models[3].pos],feed_dict = feed_dict)
                        a = np.concatenate([a,tmp_a0.reshape(-1),tmp_a1.reshape(-1),tmp_a2.reshape(-1),tmp_a3.reshape(-1)])
                #tm = time.time()
                ind = np.argsort(a[1:testset.entity_size+1])
                tmp_ind = ind
                ind = np.zeros_like(tmp_ind)
                sp1 = 0
                sp2 = len(candi_set_head[rel])
                for iii in tmp_ind:
                    if iii in candi_set_head[rel]:
                        ind[sp1] = iii
                        sp1 += 1
                    else:
                        ind[sp2] = iii
                        sp2 += 1
                
                filter = 0
                for i in xrange(testset.entity_size):
                    if (ind[i], rel, l) not in testset.ok:
                        filter += 1
                    if ind[i] == h:
                        lsum += i + 1
                        lsum_filter += filter + 1
                        lsum_r[rel] += i + 1
                        lsum_filter_r[rel] += filter + 1
                        if i < 10:
                            lp_n += 1
                            lp_n_r[rel] += 1
                        break
                
                #te = time.time()
                #print tm - ts, te -tm, te - ts 

                h_a = h * np.ones(params['batch_size']/gpu_num)
                a = np.zeros(1)
                for i in xrange(nbatches+1):
                    if i != nbatches:
                        t_a = np.arange(i*params['batch_size'],(i+1)*params['batch_size'])
                    else:
                        t_a = range(i*params['batch_size'],testset.entity_size)
                        t_a.extend(range((i+1)*params['batch_size']-testset.entity_size))
                        t_a = np.array(t_a)
                    if gpu_num == 1:
                        feed_dict = {models[0].pos_h: h_a, models[0].pos_t: t_a, models[0].rel: r_a}
                        #feed_dict = {models[0].pos_h: h_a, models[0].pos_t: t_a, models[0].rel: r_a, models[0].cls: np.zeros(params['batch_size'])}
                        tmp_a0 = sess.run(models[0].pos,feed_dict = feed_dict)
                        a = np.concatenate([a,tmp_a0.reshape(-1)])                        
                    elif gpu_num == 4:
                        feed_dict = {models[0].pos_h: h_a, models[0].pos_t: t_a[:sp[0]], models[0].rel: r_a, models[1].pos_h: h_a, models[1].pos_t: t_a[sp[0]:sp[1]], models[1].rel: r_a, models[2].pos_h: h_a, models[2].pos_t: t_a[sp[1]:sp[2]], models[2].rel: r_a, models[3].pos_h: h_a, models[3].pos_t: t_a[sp[2]:], models[3].rel: r_a}
                        tmp_a0, tmp_a1, tmp_a2, tmp_a3 = sess.run([models[0].pos, models[1].pos, models[2].pos, models[3].pos],feed_dict = feed_dict)
                        a = np.concatenate([a,tmp_a0.reshape(-1),tmp_a1.reshape(-1),tmp_a2.reshape(-1),tmp_a3.reshape(-1)])
                ind = np.argsort(a[1:testset.entity_size+1])
                tmp_ind = ind
                ind = np.zeros_like(tmp_ind)
                sp1 = 0
                sp2 = len(candi_set_tail[rel])
                for iii in tmp_ind:
                    if iii in candi_set_tail[rel]:
                        ind[sp1] = iii
                        sp1 += 1
                    else:
                        ind[sp2] = iii
                        sp2 += 1

                filter = 0
                for i in xrange(testset.entity_size):
                    if (h, rel, ind[i]) not in testset.ok:
                        filter += 1
                    if ind[i] == l:
                        rsum += i + 1
                        rsum_filter += filter + 1
                        rsum_r[rel] += i + 1
                        rsum_filter_r[rel] += filter + 1
                        if i < 10:
                            rp_n += 1
                            rp_n_r[rel] += 1
                        break

            print 'left: ', 1.*lsum/len(testset.fb_l), '\t', 1.*lp_n/len(testset.fb_l), '\t', 1.*lsum_filter/len(testset.fb_l)
            print 'right: ', 1.*rsum/len(testset.fb_r), '\t', 1.*rp_n/len(testset.fb_r), '\t', 1.*rsum_filter/len(testset.fb_l)
            if eval(params['detailed']):
                for i in xrange(testset.relation_size):
                    print testset.id2relation[i], rel_num[i]
                    print 'left: ', 1.*lsum_r[i]/rel_num[i], '\t', 1.*lp_n_r[i]/rel_num[i], '\t', 1.*lsum_filter_r[i]/rel_num[i]
                    print 'right: ', 1.*rsum_r[i]/rel_num[i], '\t', 1.*rp_n_r[i]/rel_num[i], '\t', 1.*rsum_filter_r[i]/rel_num[i]


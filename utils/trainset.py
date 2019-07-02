import os
import random
import numpy as np

import sys
sys.path.append('./')
from utils.sample import sample

class TrainSet(object):
    def __init__(self, data_path, version='unif'):
        v_arr = version.split('_')
        self.version = v_arr[0]
        # some revision for sampling
        self.center = True if 'cent' in v_arr else False
        self.asym = True if 'asym' in v_arr else False
        self.kneg = True if 'kneg' in v_arr else False

        self.entity2id = {}
        self.id2entity = []
        with open(os.path.join(data_path, 'entity2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.id2entity.append(line_arr[0])
                self.entity2id[line_arr[0]] = int(line_arr[1])

        self.relation2id = {}
        self.id2relation = []
        with open(os.path.join(data_path, 'relation2id.txt')) as fr:
            for line in fr:
                line_arr = line.strip().split()
                self.id2relation.append(line_arr[0])
                self.relation2id[line_arr[0]] = int(line_arr[1])

        self.relation_num = len(self.id2relation)
        self.entity_num = len(self.id2entity)
        set_name = data_path.split('/')[-1]
        print 'dataset {} has {} relations.'.format(set_name, self.relation_num)
        print 'dataset {} has {} entities.'.format(set_name, self.entity_num)

        self.fb_h = []
        self.fb_r = []
        self.fb_l = []
        self.ok = set()
        if self.center:
            self.r_e_h = [set() for _ in xrange(self.relation_num)]
            self.r_e_t = [set() for _ in xrange(self.relation_num)]
        if self.kneg:
            self.w_cnt = np.zeros(self.entity_num)
        if self.version == 'bern':
            left_entity = [[0 for j in xrange(self.entity_num)] for i in xrange(self.relation_num)]
            right_entity = [[0 for j in xrange(self.entity_num)] for i in xrange(self.relation_num)]
        #self.total_size = 0
        with open(os.path.join(data_path, 'train.txt')) as fr:
            for line in fr:
                #if self.total_size % 1000 == 0: print slef.total_size
                line_arr = line.strip().split()
                tmp_head = self.entity2id[line_arr[0]]
                tmp_last = self.entity2id[line_arr[1]]
                tmp_relation = self.relation2id[line_arr[2]]
                self.fb_h.append(tmp_head)
                self.fb_l.append(tmp_last)
                self.fb_r.append(tmp_relation)
                self.ok.add((tmp_head, tmp_relation, tmp_last))
                if self.center:
                    self.r_e_h[tmp_relation].add(tmp_head)
                    self.r_e_t[tmp_relation].add(tmp_last)
                if self.kneg:
                    self.w_cnt[tmp_head] += 1
                    self.w_cnt[tmp_last] += 1
                if self.version == 'bern':
                    left_entity[tmp_relation][tmp_head] += 1
                    right_entity[tmp_relation][tmp_last] += 1
        
                #self.total_size += 1
        
        self.head_set = [set() for _ in xrange(self.relation_num)]
        self.tail_set = [set() for _ in xrange(self.relation_num)]
        for item in zip(self.fb_h,self.fb_r,self.fb_l):
            self.head_set[item[1]].add(item[0])
            self.tail_set[item[1]].add(item[2])
        if self.center:
            self.r_e_h = [list(item) for item in self.r_e_h]
            self.r_e_t = [list(item) for item in self.r_e_t]
        if self.kneg:
            self.w_cnt = self.w_cnt / np.sum(self.w_cnt)
        if self.version == 'bern':
            self.r_p = []
            for i in xrange(self.relation_num):
                suml1 = 0; suml2 = 0
                sumr1 = 0; sumr2 = 0
                for j in xrange(self.entity_num):
                    if left_entity[i][j] != 0:
                        suml1 += 1
                        suml2 += left_entity[i][j]
                    if right_entity[i][j] != 0:
                        sumr1 += 1
                        sumr2 += right_entity[i][j]
                lnum = 1.*suml2/suml1
                rnum = 1.*sumr2/sumr1
                self.r_p.append(rnum/(rnum+lnum))    

        self.record_point = 0
        self.record_num = len(self.fb_h)

    def batch_gen(self, batch_size):
        while True:
            if self.record_point == 0:
                self.entity1 = np.zeros(self.record_num, dtype = np.int32)
                self.entity2 = np.zeros(self.record_num, dtype = np.int32)
                self.entity3 = np.zeros(self.record_num, dtype = np.int32)
                self.entity4 = np.zeros(self.record_num, dtype = np.int32)
                self.relation = np.zeros(self.record_num, dtype = np.int32)
                index = range(self.record_num)
                random.shuffle(index)
                kneg_ind = 0
                if self.kneg:
                    smp_ind = sample(2*self.record_num, self.w_cnt)
                for i in index:
                    if self.kneg:
                        j = smp_ind[kneg_ind % (2*self.record_num)]
                        kneg_ind += 1
                    else:
                        j = random.randint(0,self.entity_num-1)
                    if self.version == 'bern':
                        pr = self.r_p[self.fb_r[i]]
                    else:
                        pr = 0.5
                    tmp_r = self.fb_r[i]
                    if random.random() < pr:
                        if self.center and random.random() < 0.4:
                            j = j % len(self.r_e_t[tmp_r])
                            j = self.r_e_t[tmp_r][j]
                        elif self.asym and random.random() < 0.05:
                            j = self.fb_h[i]
                        while (self.fb_h[i], self.fb_r[i], j) in self.ok:
                            if self.center and random.random() < 0.4:
                                j = random.randint(0,len(self.r_e_t[tmp_r])-1)
                                j = self.r_e_t[tmp_r][j]
                            else:
                                if self.kneg:
                                    j = smp_ind[kneg_ind % (2*self.record_num)]
                                    kneg_ind += 1
                                else:
                                    j = random.randint(0,self.entity_num-1)
                        self.entity1[i] = self.fb_h[i]
                        self.entity2[i] = self.fb_l[i]
                        self.entity3[i] = self.fb_h[i]
                        self.entity4[i] = j
                        self.relation[i] = self.fb_r[i]
                    else:
                        if self.center and random.random() < 0.4:
                            j = j % len(self.r_e_h[tmp_r])
                            j = self.r_e_h[tmp_r][j]
                        elif self.asym and random.random() < 0.05:
                            j = self.fb_l[i]
                        while (j, self.fb_r[i], self.fb_l[i]) in self.ok:
                            if self.center and random.random() < 0.4:
                                j = random.randint(0,len(self.r_e_h[tmp_r])-1)
                                j = self.r_e_h[tmp_r][j]
                            else:
                                if self.kneg:
                                    j = smp_ind[kneg_ind % (2*self.record_num)]
                                    kneg_ind += 1
                                else:
                                    j = random.randint(0,self.entity_num-1)
                        self.entity1[i] = self.fb_h[i]
                        self.entity2[i] = self.fb_l[i]
                        self.entity3[i] = j
                        self.entity4[i] = self.fb_l[i]
                        self.relation[i] = self.fb_r[i]
            
            end = self.record_point + batch_size        
            '''
            if end < self.record_num:
                ind = range(self.record_point,end)
                self.record_point += batch_size
            else:
                ind = range(self.record_point,self.record_num)
                ind.extend(range(end-self.record_num))
                self.record_point = 0
            '''
            ind = range(self.record_point,end)
            self.record_point += batch_size
            if end + batch_size > self.record_num:
                self.record_point = 0
            entity1 = self.entity1[ind]
            entity2 = self.entity2[ind]
            entity3 = self.entity3[ind]
            entity4 = self.entity4[ind]
            relation = self.relation[ind]
            
            yield entity1, entity2, entity3, entity4, relation

import os
import random
import numpy as np

import sys
sys.path.append('./')
from utils.sample import sample

data_path = './data/FB15k'
model_path = ''

entity2id = {}
id2entity = []
with open(os.path.join(data_path, 'entity2id.txt')) as fr:
    for line in fr:
        line_arr = line.strip().split()
        id2entity.append(line_arr[0])
        entity2id[line_arr[0]] = int(line_arr[1])

relation2id = {}
id2relation = []
with open(os.path.join(data_path, 'relation2id.txt')) as fr:
    for line in fr:
        line_arr = line.strip().split()
        id2relation.append(line_arr[0])
        relation2id[line_arr[0]] = int(line_arr[1])

relation_num = len(id2relation)
entity_num = len(id2entity)
set_name = data_path.split('/')[-1]
print 'load id done.'
print 'dataset {} has {} relations.'.format(set_name, relation_num)
print 'dataset {} has {} entities.'.format(set_name, entity_num)


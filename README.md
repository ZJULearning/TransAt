TransAt:Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism
======

TransAt is a translation based embedding model for Knowledge Graph Completion. It implements the algorithm of our IJCAI2018 paper: [Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism](https://www.ijcai.org/proceedings/2018/0596.pdf) 

Benchmark datasets
------

* [FB15k,WN18](https://everest.hds.utc.fr/doku.php?id=en:transe)
* [FB13,WN11](http://cs.stanford.edu/~danqi/data/nips13-dataset.tar.bz2)

Datasets are required in the folder data/ in the following format, containing five files:

+ train.txt: training file, format (e1, e2, rel).

+ valid.txt: validation file, same format as train.txt

+ test.txt: test file, same format as train.txt.

+ entity2id.txt: all entities and corresponding ids, one per line.

+ relation2id.txt: all relations and corresponding ids, one per line.

Link prediction performance on WN18
------

| Model      |     MeanRank(Raw) |   MeanRank(Filter)   |	Hit@10(Raw)	| Hit@10(Filter)|
| :-------- | --------:| :------: | :------: |:------: |
| TransE            |    263  |     251 |      75.4 |       89.2|
| TransH(unif/bern) | 318/401 | 303/388 | 75.4/73.0 | 86.7/82.3 |
| TransR(unif/bern) | 232/238 | 219/225 | 78.3/79.8 | 91.7/92.0 |
|CTransR (unif/bern)| 243/231 | 230/218 | 78.9/79.4 | 92.3/92.3 |
|TransD (unif/bern) | 242/224 | 229/212 | 79.2/79.6 | 92.5/92.2 |
|TranSparse (share, S, unif/bern) | 248/237 | 236/224 | 79.7/80.4 | 93.5/93.6 |
|TranSparse (share, US, unif/bern)| 242/233 | 229/221 | 79.8/80.5 | 93.7/93.9 |
|TranSparse (separate, S, unif/bern)| 235/224 |223/221 | 79.0/79.8 | 92.3/92.8 |
|TranSparse (separate, US, unif/bern)| 233/223 | 221/211 | 79.6/80.1 | 93.4/93.2 |
|TransAt (bern)| 214 | 202 | 81.4 | 95.1 |
|TransAt (asy,bern)| 169 | 157 | 81.4 | 95.0 |

How to use (require tensorflow 1.1.0 and python 2.7 with numpy, sklearn, cPickle)
------

train on WN18:
1. change "phase" variable in conf/TransAll_v1_WN18.cfg to be "train".
2. run "./scripts/TransAll_v1/TransAll_v1_WN18.sh"
test on WN18:
1. change "phase" variable in conf/TransAll_v1_WN18.cfg to be "test".
2. run "./scripts/TransAll_v1/TransAll_v1_WN18.sh"

Reference
------

Reference to cite when you use TransAt in a research paper
```
@inproceedings{qian2018translating,
  title={Translating Embeddings for Knowledge Graph Completion with Relation Attention Mechanism.},
  author={Qian, Wei and Fu, Cong and Zhu, Yu and Cai, Deng and He, Xiaofei},
  booktitle={IJCAI},
  pages={4286--4292},
  year={2018}
```

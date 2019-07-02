import numpy as np
import random
import time

def sample(m, prob):
    '''
    Probability sampling with replacement.
    Inputs:
        m: expected sample number with int type
        prob: sample distribution with (n,) array type
    Returns:
        s_id: a list of sampled id
    '''
    if type(np.array(prob)) != np.ndarray:
        raise ValueError('prob need to be array type.')
    if np.any(prob < 0) or not (0.999 <= np.sum(prob) <= 1.001):
        raise ValueError('prob need to be a probability distribution.')

    n = len(prob)
    nc = 0
    for i in xrange(n):
        if n*prob[i] > 0.1:
            nc += 1
    if nc > 200:
        return AliasTable_ProbSampleReplace(n, m, prob)
    else:
        return ProbSampleReplace(n, m, prob)

def ProbSampleReplace(n, m, prob):
    '''
    Unequal probability sampling; with-replacement case
    '''
    perm = np.argsort(prob)
    p = np.cumsum(prob[perm[::-1]])

    s_id = []
    for i in xrange(m):
        tmp = random.random()
        for j in xrange(n):
            if tmp < p[j]:
                break
        s_id.append(perm[n-j-1])
    
    return s_id

def AliasTable_ProbSampleReplace(n, m, prob):
    q = prob * n
    H = -1
    L = n
    HL = np.zeros(n,dtype=np.int32)
    a = np.zeros(n,dtype=np.int32)
    for i in xrange(n):
        if q[i] < 1.:
            H += 1
            HL[H] = i
        else:
            L -= 1
            HL[L] = i
    if H >= 0 and L < n:
        for k in xrange(n-1):
            i = HL[k]
            j = HL[L]
            a[i] = j;
            q[j] += q[i] - 1;
            if q[j] < 1.:
                L += 1
            if L >= n:
                break
    for i in xrange(n):
        q[i] += i

    s_id = []
    for i in xrange(m):
        tmp = random.random() * n
        k = int(tmp)
        s_id.append(k if tmp < q[k] else a[k])

    return s_id

if __name__ == "__main__":
    # example 1
    prob = np.ones(1000) / 1000.
    m = 100
    ind = sample(m, prob)
    # 100 is right
    print len(ind)
    # example 2
    prob = np.zeros(1000)
    for i in xrange(1000):
        if 0 <= i % 10 <= 6:
            prob[i] = 0.3*(1./700)
        else:
            prob[i] = 0.7*(1./300)
    m = 100
    cnt = 0
    t = time.time()
    for i in xrange(100):
        ind = sample(m, prob)
        for item in ind:
            if 0 <= item%10 <= 6:
                cnt += 1
    # 30 is right
    print cnt/100, time.time()-t
    cnt = 0
    t = time.time()
    for i in xrange(100):
        ind = ProbSampleReplace(1000, m, prob)
        for item in ind:
            if 0 <= item%10 <= 6:
                cnt += 1
    # 30 is right
    print cnt/100, time.time()-t

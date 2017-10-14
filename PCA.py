# -*- coding: utf-8 -*-

'''
主成分分析モジュール
'''

import numpy as np

def PC(X):
    '''
    主成分を求める関数
    @param X データ行列
    @return 主成分行列
    '''
    Xc = X-X.mean(axis=0, keepdims=True)
    U, s, VT = np.linalg.svd(Xc, full_matrices=False)
    return U*s[np.newaxis, :]

def kPC(K):
    '''
    カーネル主成分を求める関数
    @param K グラム行列
    @return カーネル主成分行列
    '''
    Kc = K-K.mean(axis=0, keepdims=True)-K.mean(axis=1, keepdims=True)+K.mean()
    l, U = np.linalg.eigh(Kc)
    return (U*np.sqrt(l)[np.newaxis, :])[:, ::-1]

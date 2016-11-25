# -*- coding: utf-8 -*-

'''
# digipictモジュール
デジタル図形を提供するモジュール
'''

import numpy as np

def digiline(P, Q):
    '''
    デジタル線分関数
    @param P: (2, ) 始点
    @param Q: (2, ) 終点
    @return np.array((L, 2)) 標本点たち
    '''
    if abs(Q[0]-P[0]) > abs(Q[1]-P[1]):
        # 寝た線分
        if P[0] > Q[0]:
            P, Q = Q, P
        x1, y1 = P.tolist()
        col1, row1 = P.astype(int).tolist()
        x2, y2 = Q.tolist()
        col2, row2 = Q.astype(int).tolist()

        cols = np.arange(col1, col2+1)
        rows = ((cols+1/2)*(y2-y1)/(x2-x1)+(x2*y1-x1*y2)/(x2-x1)).astype(int)
        rows[0], rows[-1] = row1, row2
    else:
        # 立った線分
        if P[1] > Q[1]:
            P, Q = Q, P
        x1, y1 = P.tolist()
        col1, row1 = P.astype(int).tolist()
        x2, y2 = Q.tolist()
        col2, row2 = Q.astype(int).tolist()

        rows = np.arange(row1, row2+1)
        cols = ((rows+1/2)*(x2-x1)/(y2-y1)+(x1*y2-x2*y1)/(y2-y1)).astype(int)
        cols[0], cols[-1] = col1, col2

    return np.stack((cols, rows), axis=1)


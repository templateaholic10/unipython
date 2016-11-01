# -*- coding: utf-8 -*-

"""
numpyを補助するモジュール
"""

import numpy as np

def mxvs(M, vs):
    """
    行列とベクトル列の積
    @param M 行列
    @param vs ベクトル列
    @return ベクトル列
    """
    return np.tensordot(vs, M, axes=(1, 1))

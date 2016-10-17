# -*- coding: utf-8 -*-

"""
# gfourierモジュール

グラフフーリエ変換（Graph Fourier Transform, GFT）モジュール．
グラフはnetworkx.Graphで表現する．
"""

import numpy as np
import numpy.linalg as LA
import networkx as nx

class GFT:
    """
    グラフフーリエ変換クラス
    """
    def __init__(self, G):
        """
        コンストラクタ
        @param G networkx.Graph
        """
        self.G = G
        self.N = len(G.nodes())
        self.L = nx.laplacian_matrix(G).toarray()
        self.l, self.U = LA.eigh(self.L)

    def gft(self, f, axis=None):
        """
        グラフフーリエ変換
        @param f グラフ信号ベクトル，またはその配列
        @param axis ベクトルの軸
        @return グラフスペクトルベクトル，またはその配列
        """
        if axis is None:
            axis = len(f.shape)-1
        return np.tensordot(f, self.U.T.conj(), axes=(axis, 1))

    def igft(self, F, axis=None):
        """
        逆グラフフーリエ変換
        @param F グラフスペクトルベクトル，またはその配列
        @param axis ベクトルの軸
        @return グラフ信号ベクトル，またはその配列
        """
        if axis is None:
            axis = len(F.shape)-1
        return np.tensordot(F, self.U, axes=(axis, 1))

    def T(self, f, u):
        """
        頂点シフト演算子
        @param f グラフ信号ベクトル
        @param u シフト先の頂点
        @return グラフ信号ベクトル
        """
        return np.sqrt(self.N) * self.igft(self.gft(f) * self.U[u])

    def M(self, f, z):
        """
        振幅変調演算子
        @param f グラフ信号ベクトル
        @param z 何番目の周波数に変調するか
        @return グラフ信号ベクトル
        """
        return np.sqrt(self.N) * f * self.U[:, z]

class WGFT(GFT):
    """
    窓付きグラフフーリエ変換クラス
    """
    def __init__(self, G, g):
        """
        コンストラクタ
        @param G networkx.Graph
        @param g グラフ窓ベクトル
        """
        self.g = g
        GFT.__init__(self, G)

        # 局在化ベクトルのノルム（||T[n]g||）
        # グラフフーリエ原子（g[u][k][n]）
        self.Tgnorms = np.zeros(self.N)
        self.atoms = np.zeros((self.N, self.N, self.N))
        for u in range(self.N):
            Tg = GFT.T(self, g, u)
            self.Tgnorms[u] = LA.norm(Tg)
            for k in range(self.N):
                MTg = GFT.M(self, Tg, k)
                self.atoms[u, k] = MTg

    def wgft(self, f):
        """
        窓付きフーリエ変換
        @param f グラフ信号ベクトル
        @return Sf (頂点, 周波数)成分からなる行列
        """
        Sf = np.zeros((self.N, self.N))
        for u in range(self.N):
            for k in range(self.N):
                Sf[u, k] = np.dot(f, self.atoms[u, k])
        return Sf

    def iwgft(self, Sf):
        """
        窓付きフーリエ逆変換
        @param Sf (頂点, 周波数)成分からなる行列
        @return f グラフ信号ベクトル
        """
        f = np.zeros(self.N)
        for u in range(self.N):
            for k in range(self.N):
                f += Sf[u, k]*self.atoms[u, k]
        f /= self.N*self.TGnorms**2
        return f

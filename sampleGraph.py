# -*- coding: utf-8 -*-

"""
# sampleGraphモジュール

networkx.Graphのサンプルを生成する関数を収めたモジュール．
"""

import numpy as np
import numpy.linalg as LA
import networkx as nx
import itertools as it

def naphthalene():
    """
    ナフタレングラフを作る関数
    @return G, pos networkx.Graphとレイアウト
    """

    # グラフの作成
    N = 10
    G = nx.Graph()
    nodes = list(range(N))
    G.add_nodes_from(nodes)
    edges = [(i, i+1) for i in range(N-1)] + [(N-1, 0)] + [(0, round(N/2))]
    G.add_edges_from(edges)

    # 座標の作成
    sq3 = np.sqrt(3)
    pos = dict()
    pos[0] = (2*sq3, 1)
    pos[1] = (3*sq3, 0)
    pos[2] = (4*sq3, 1)
    pos[3] = (4*sq3, 3)
    pos[4] = (3*sq3, 4)
    pos[5] = (2*sq3, 3)
    pos[6] = (sq3, 4)
    pos[7] = (0, 3)
    pos[8] = (0, 1)
    pos[9] = (sq3, 0)

    return G, pos

def random_sensor_network(N=500, sigma1=0.074, sigma2=0.075):
    """
    random sensor networkを生成する関数．
    正方形の上にn点を配置し，ユークリッド距離に基づくガウシアン重み関数で重み付ける
    @param N 頂点の数
    @param sigma1 重みを大きくする
    @param sigma2 辺で結ばれやすくする
    @return G, pos netoworkx.Graphとレイアウト
    """

    # ノードの作成
    G = nx.Graph()
    nodes = list(range(N))
    G.add_nodes_from(nodes)

    # 単位正方形にばらまいてガウシアン重み関数で重み付ける
    pos_array = np.random.rand(N, 2)
    pos = {n: (elem[0], elem[1]) for n, elem in enumerate(pos_array)}
    for pair in it.combinations(nodes, 2):
        u, v = pair
        d = LA.norm(pos_array[u] - pos_array[v])
        if d <= sigma2:
            G.add_edge(u, v, weight=np.exp(-d**2/(2*sigma1**2)))

    return G, pos

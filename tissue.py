# -*- coding: utf-8 -*-

'''
# tissueモジュール
充填的グラフ埋め込み（外周にのみ次数1の頂点を許すグラフ埋め込み）に関する操作を提供するモジュール
'''

import numpy as np
import networkx as nx
import digipict

def cells(G, pos):
    '''
    充填的グラフ埋め込みの閉じたセルと開いたセルを列挙する関数
    @param G networkx.Graph
    @param pos 頂点の座標
    @return 閉じたセルと開いたセルのタプル．セルは反時計回りの頂点番号リストで表される
    '''
    H = G.to_directed()
    nx.set_edge_attributes(G, 'passed', False)

    # 開いたセルの列挙
    opened = []

    if 1 in list(G.degree().values()):
        # 外周セルがある場合
        u = u0 = list(G.degree().values()).index(1) # 適当な出発点
        v = list(G[u].keys())[0]
        H[u][v]['passed'] = True
        vertexes = [u, v]

        while v != u0:
            # スタート地点に戻るまでループ
            if len(G[v]) == 1:
                # 現在地の次数1ならばセルを独立させる
                u, v = v, u
                opened.append(vertexes)
                vertexes = [u, v]
                H[u][v]['passed'] = True
            else:
                # 現在地の次数1でなければセルを延長する

                # 反時計回りに次の頂点を探索する
                AB = pos[u]-pos[v]
                theta = np.arctan2(AB[1], AB[0])
                AC = pos[list(G[v].keys())]-pos[v]
                thetas = np.arctan2(AC[:, 1], AC[:, 0])
                next_id = np.mod(thetas-theta, 2*np.pi).argmax()

                u = v
                v = list(G[v].keys())[next_id]
                vertexes.append(v)
                H[u][v]['passed'] = True
        else:
            opened.append(vertexes)
            vertexes = []

    # 閉じたセルの列挙
    closed = []

    while True:
        # 通っていない辺を探索する
        for u, v, passed in H.edges_iter(data='passed'):
            if not passed:
                u0, v0 = u, v
                break
        else:
            break

        u, v = u0, v0
        H[u][v]['passed'] = True
        vertexes = [u, v]

        while True:
            # 反時計回りに次の頂点を探索する
            AB = pos[u]-pos[v]
            theta = np.arctan2(AB[1], AB[0])
            AC = pos[list(G[v].keys())]-pos[v]
            thetas = np.arctan2(AC[:, 1], AC[:, 0])
            next_id = np.mod(thetas-theta, 2*np.pi).argmax()

            u = v
            v = list(G[v].keys())[next_id]
            H[u][v]['passed'] = True
            if v == u0:
                break
            vertexes.append(v)

        closed.append(vertexes)
        vertexes = []

    return closed, opened

def polyvolume(pos):
    '''
    閉じたセルの符号つき面積を求める関数
    @param pos: np.array((V, 2)) 閉路の頂点の座標
    @return 反時計回りを正とする符号つき面積
    '''
    xs, ys = pos[:, 0], pos[:, 1]
    return (np.roll(xs, 1)-xs).dot(np.roll(ys, 1)+ys)/2

def openpolyvolume(pos, background):
    '''
    開いたセルの面積を求める関数
    @param pos: np.array((V, 2)) パスの頂点の座標
    @param background: np.array((N, M), dtype=bool) 内部Trueと外部Falseからなるbool平面
    @return 面積
    '''
    canvas = np.full_like(background, 0, dtype=np.int8)
    Rows, Cols = canvas.shape
    stack = []
    V, _ = pos.shape

    # 最初と最後の多角辺はキャンバスの端まで非活性化する
    eps = 1e-6

    P, Q = pos[1], pos[0]
    dx, dy = (Q-P).tolist()
    if dx >= 0 and dy >= 0:
        if dx/(Cols-P[0]) > dy/(Rows-P[1]):
            R = np.array((Cols-eps, P[1]+(Cols-eps-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(Rows-eps-P[1])*dx/dy, Rows-eps))
    elif dx >= 0 and dy < 0:
        if dx/(Cols-P[0]) > dy/(0-P[1]):
            R = np.array((Cols-eps, P[1]+(Cols-eps-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(0-P[1])*dx/dy, 0))
    elif dx <  0 and dy >= 0:
        if dx/(0-P[0]) > dy/(Rows-P[1]):
            R = np.array((0, P[1]+(0-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(Rows-eps-P[1])*dx/dy, Rows-eps))
    else:
        if dx/(0-P[0]) > dy/(0-P[1]):
            R = np.array((0, P[1]+(0-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(0-P[1])*dx/dy, 0))

    line = digipict.digiline(P, R)
    if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
        line = line[1:]
    if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
        line = line[:-1]
    canvas[line[:, 1], line[:, 0]] = -1

    P, Q = pos[-2], pos[-1]
    dx, dy = (Q-P).tolist()
    if dx >= 0 and dy >= 0:
        if dx/(Cols-P[0]) > dy/(Rows-P[1]):
            R = np.array((Cols-eps, P[1]+(Cols-eps-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(Rows-eps-P[1])*dx/dy, Rows-eps))
    elif dx >= 0 and dy < 0:
        if dx/(Cols-P[0]) > dy/(0-P[1]):
            R = np.array((Cols-eps, P[1]+(Cols-eps-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(0-P[1])*dx/dy, 0))
    elif dx <  0 and dy >= 0:
        if dx/(0-P[0]) > dy/(Rows-P[1]):
            R = np.array((0, P[1]+(0-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(Rows-eps-P[1])*dx/dy, Rows-eps))
    else:
        if dx/(0-P[0]) > dy/(0-P[1]):
            R = np.array((0, P[1]+(0-P[0])*dy/dx))
        else:
            R = np.array((P[0]+(0-P[1])*dx/dy, 0))

    line = digipict.digiline(P, R)
    if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
        line = line[1:]
    if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
        line = line[:-1]
    canvas[line[:, 1], line[:, 0]] = -1

    # 多角辺上の点を非活性化する．中点の内側をシードとして活性化する
    for v in range(V-1):
        line = digipict.digiline(pos[v], pos[v+1])
        # デジタル線分が線分の外に出るとき，端点がキャンバス外に出る場合がある
        if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
            line = line[1:]
        if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
            line = line[:-1]
        canvas[line[:, 1], line[:, 0]] = -1
        seed = line[len(line)//2]+np.array((1 if pos[v][1]>pos[v+1][1] else -1, 1 if pos[v][0]<pos[v+1][0] else -1))
        canvas[seed[1], seed[0]] = 1
        stack.append(seed)

    offsets = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)))
    while stack:
        nbrs = stack.pop() + offsets # 4 neighbors
        nbrs = nbrs[(nbrs[:, 0] >= 0) & (nbrs[:, 0] < Cols) & (nbrs[:, 1] >= 0) & (nbrs[:, 1] < Rows)]
        homs = nbrs[(canvas[nbrs[:, 1], nbrs[:, 0]] == 0) & (background[nbrs[:, 1], nbrs[:, 0]])]

        canvas[homs[:, 1], homs[:, 0]] = 1
        stack += [hom for hom in homs]

    return (canvas == 1).sum()

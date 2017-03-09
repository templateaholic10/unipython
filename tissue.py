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

def polyvol(pos):
    '''
    閉じたセルの符号つき面積を求める関数
    @param pos: np.array((V, 2)) 閉路の頂点の座標
    @return 反時計回りを正とする符号つき面積
    '''
    xs, ys = pos[:, 0], pos[:, 1]
    return (np.roll(xs, 1)-xs).dot(np.roll(ys, 1)+ys)/2

def cpclock(pos, needle):
    '''
    凸多角形時計の時刻を読む関数
    @param pos 凸多角形の反時計回りの頂点座標
    @param needle 時計の針の変位
    @return 時刻
    '''
    nvec = np.asarray((-needle[1],needle[0]))
    # Q-[V-1,0,...,V-2]とQPの法線の内積が0以下 and Q-[0,1,...,V-1]とQPの法線の内積が0以上
    # = 反時計回りに0以下から0以上に切り替わる辺
    # ともに0を含めないと，角で凸多角形と接するときに時刻が存在しない場合がある
    tf = np.logical_and(np.roll(pos.dot(nvec),1)<=0, pos.dot(nvec)>=0)
    index = tf.argmax()
    return index if tf[index] else None

def __xic(P, Q, y):
    '''
    x切片を求める関数
    @param P 制御点１
    @param Q 制御点２
    @param y y座標
    @return x切片
    '''
    return (Q[0]-P[0])/(Q[1]-P[1])*(y-P[1])+P[0]

def __yic(P, Q, x):
    '''
    y切片を求める関数
    @param P 制御点１
    @param Q 制御点２
    @param x x座標
    @return y切片
    '''
    return (Q[1]-P[1])/(Q[0]-P[0])*(x-P[0])+P[1]

def rectclock_ic(xlim, ylim, center, needle):
    '''
    矩形時計の時刻を読み，交点を求める関数
    @param xlim x座標の閉区間
    @param ylim y座標の閉区間
    @param center 時計の中心座標
    @param needle 時計の針の変位
    @return 時刻と交点のタプル．交点がないとき，時刻は定義されず，ともにNoneを返す
    '''
    rect = np.asarray(((xlim[0],ylim[0]),(xlim[1],ylim[0]),(xlim[1],ylim[1]),(xlim[0],ylim[1])))
    edge = cpclock(rect-center,needle)
    if edge is None:
        Pic = None
    elif needle[0]<0 and needle[1]==0:
        Pic = np.asarray((xlim[0],center[1])) # ←
    elif needle[0]==0 and needle[1]<0:
        Pic = np.asarray((center[0],ylim[0])) # ↓
    elif needle[0]>0 and needle[1]==0:
        Pic = np.asarray((xlim[1],center[1])) # →
    elif needle[0]==0 and needle[1]>0:
        Pic = np.asarray((center[0],ylim[1])) # ↑
    elif edge==0:
        Pic = np.asarray((xlim[0],__yic(center,center+needle,xlim[0]))) # ←
    elif edge==1:
        Pic = np.asarray((__xic(center,center+needle,ylim[0]),ylim[0])) # ↓
    elif edge==2:
        Pic = np.asarray((xlim[1],__yic(center,center+needle,xlim[1]))) # →
    else:
        Pic = np.asarray((__xic(center,center+needle,ylim[1]),ylim[1])) # ↑
    return edge, Pic

def fill(canvas, boundary=None):
    '''
    塗りつぶし関数
    @param canvas: np.array((Rows, Cols), dtype=bool) キャンバスの定義関数
    @param boundary: np.array((V, 2)) 区分的に直線になる境界．境界を反時計回りに見る側が内側
    @return 面積
    '''
    canvas_tmp = np.full_like(canvas, -1, dtype=np.int8)
    canvas_tmp[canvas] = 0
    Rows, Cols = canvas.shape
    stack = []
    V, _ = boundary.shape

    # 多角辺上の点を非活性化する．中点の内側をシードとして活性化する
    for v in range(V-1):
        line = digipict.digiline(boundary[v], boundary[v+1])
        if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
            line = line[1:] # 始点が矩形外の時，切り落とす
        if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
            line = line[:-1] # 終点が矩形外の時，切り落とす
        canvas_tmp[line[:, 1], line[:, 0]] = -1
        seed = line[len(line)//2]+np.array((1 if boundary[v][1]>boundary[v+1][1] else -1, 1 if boundary[v][0]<boundary[v+1][0] else -1))
        canvas_tmp[seed[1], seed[0]] = 1
        stack.append(seed)

    # 最初と最後の多角辺はキャンバスの端まで非活性化する
    eps = 1e-6

    _, pos_ex0 = rectclock_ic((0, Cols-eps), (0, Rows-eps), boundary[1], boundary[0]-boundary[1])
    line = digipict.digiline(boundary[0], pos_ex0)
    if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
        line = line[1:] # 始点が矩形外の時，切り落とす
    if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
        line = line[:-1] # 終点が矩形外の時，切り落とす
    canvas_tmp[line[:, 1], line[:, 0]] = -1

    _, pos_exn1 = rectclock_ic((0, Cols-eps), (0, Rows-eps), boundary[-2], boundary[-1]-boundary[-2])
    line = digipict.digiline(boundary[-1], pos_exn1)
    if not(0 <= line[0][0] < Cols and 0 <= line[0][1] < Rows):
        line = line[1:] # 始点が矩形外の時，切り落とす
    if not(0 <= line[-1][0] < Cols and 0 <= line[-1][1] < Rows):
        line = line[:-1] # 終点が矩形外の時，切り落とす
    canvas_tmp[line[:, 1], line[:, 0]] = -1

    offsets = np.array(((0, -1), (0, 1), (-1, 0), (1, 0)))
    while stack:
        nbrs = stack.pop() + offsets # 4 neighbors
        nbrs = nbrs[(nbrs[:, 0] >= 0) & (nbrs[:, 0] < Cols) & (nbrs[:, 1] >= 0) & (nbrs[:, 1] < Rows)]
        actives = nbrs[canvas_tmp[nbrs[:, 1], nbrs[:, 0]] == 0]
        canvas_tmp[actives[:, 1], actives[:, 0]] = 1
        stack += list(actives)

    return canvas_tmp

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

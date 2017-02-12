# -*- coding: utf-8 -*-

'''
標本点上の関数の積分を提供するモジュール．
標本点は半整数格子上にあるとする．
例えば，領域[0, 1)x[0, 1)には標本点(1/2, 1/2)が対応する．
'''

import numpy as np
import networkx as nx
import scipy.optimize as opt

def line_integral2d_0(f, start, end):
    '''
    ２次元標本点上のスカラー場を線分に沿って線積分する関数
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param start: np.ndarray(2) 始点
    @param end: np.ndarray(2) 終点
    @return 線積分 \int_{start->end} f dl
    '''
    # 準備

    if abs(end[0]-start[0]) >= abs(end[1]-start[1]):
        # 寝た線分ならば，始点を左にしておく
        if start[0] > end[0]:
            start, end = end, start
    else:
        # 立った線分ならば，始点を下にしておく
        if start[1] > end[1]:
            start, end = end, start

    x0, y0 = start.tolist() # 始点
    x1, y1 = end.tolist() # 終点
    col0, row0 = start.astype(int).tolist() # 始点の属するセル
    col1, row1 = end.astype(int).tolist() # 終点の属するセル

    # ４つの場合に分けて計算する

    if y0 == y1:
        # (1)x軸に平行な線分．線分を含むセルの値を足し上げて端点の外側を引く
        return f[row0, col0:col1+1].sum() - (x0-col0)*f[row0, col0] - (col1+1-x1)*f[row0, col1]

    elif x0 == x1:
        # (2)y軸に平行な線分．線分を含むセルの値を足し上げて端点の外側を引く
        return f[row0:row1+1, col0].sum() - (y0-row1)*f[row0, col0] - (row1+1-y1)*f[row1, col0]

    elif abs(x1-x0) >= abs(y1-y0):
        # (3)寝た線分
        dydx = (y1-y0)/(x1-x0) # 傾きdy/dx
        yic = y0-dydx*x0 # y切片

        rowB, rowT = sorted([row0, row1]) # 最下行と最上行
        xs_x_H = (np.arange(rowB+1, rowT+1)-yic)/dydx # 水平線との交点のx座標

        # (3-1)水平線との交点を含む列
        # 列内で２行にまたがっているので，２つのセルでの値の凸結合を加算する
        cols_x_H = xs_x_H.astype(int) # 水平線との交点を含む列のインデックス
        ts = xs_x_H-cols_x_H # 水平線との交点のx座標の垂直線からの変位
        if y0 <= y1:
            # 始点が下の場合
            integral_x_H = (ts*f[np.arange(rowB, rowT), cols_x_H]+(1-ts)*f[np.arange(rowB+1, rowT+1), cols_x_H]).sum()
        else:
            # 始点が上の場合
            integral_x_H = (ts*f[np.arange(rowB+1, rowT+1), cols_x_H]+(1-ts)*f[np.arange(rowB, rowT), cols_x_H]).sum()

        # (3-2)水平線との交点を含まない列
        # 列内で１行しか通らないので，１つのセルでの値を加算する
        cols_nox_H = np.setdiff1d(np.arange(col0, col1+1), cols_x_H) # 水平線との交点を含まない列のインデックス
        rows = (dydx*cols_nox_H+yic).astype(int) # 各列内で通る行のインデックス．１行しか通らないので，例えば列の左端との交点を含む行とすれば良い
        # ただし，交点を含まない最左の列が線分の最左列であって
        # 1. 線分は列内で水平線をまたがないが
        # 2. 線分を延長すると列内で水平線をまたぐ
        # 時，列の左端との交点を含む行と線分の通る行は異なる．
        # この場合は線分の通る行を左端点の属する行とする
        if len(cols_nox_H) > 0 and cols_nox_H[0] == col0:
            rows[0] = row0
        integral_nox_H = f[rows, cols_nox_H].sum()

        # (3-3)端の列で足しすぎた分
        extraintegral = (x0-col0)*f[row0, col0]+(col1+1-x1)*f[row1, col1]

        return (integral_x_H+integral_nox_H-extraintegral)/(x1-x0)*np.linalg.norm(end-start)

    else:
        # (4)立った線分
        dxdy = (x1-x0)/(y1-y0) # 傾きdx/dy
        xic = x0-dxdy*y0 # x切片

        colL, colR = sorted([col0, col1]) # 最左列と最右列
        ys_x_V = (np.arange(colL+1, colR+1)-xic)/dxdy # 垂直線との交点のy座標

        # (4-1)垂直線との交点を含む行
        # 行内で２列にまたがっているので，２つのセルでの値の凸結合を加算する
        rows_x_V = ys_x_V.astype(int) # 垂直線との交点を含む行のインデックス
        ts = ys_x_V-rows_x_V # 垂直線との交点のy座標の水平線からの変位
        if x0 <= x1:
            # 始点が左の場合
            integral_x_V = (ts*f[rows_x_V, np.arange(colL, colR),]+(1-ts)*f[rows_x_V, np.arange(colL+1, colR+1)]).sum()
        else:
            # 始点が右の場合
            integral_x_V = (ts*f[rows_x_V, np.arange(colL+1, colR+1)]+(1-ts)*f[rows_x_V, np.arange(colL, colR)]).sum()

        # (4-2)垂直線との交点を含まない行
        # 行内で１列しか通らないので，１つのセルでの値を加算する
        rows_nox_V = np.setdiff1d(np.arange(row0, row1+1), rows_x_V) # 垂直線との交点を含まない行のインデックス
        cols = (dxdy*rows_nox_V+xic).astype(int) # 各行内で通る列のインデックス．１列しか通らないので，例えば行の下端との交点を含む列とすれば良い
        # ただし，交点を含まない最下の列が線分の最下列であって
        # 1. 線分は行内で垂直線をまたがないが
        # 2. 線分を延長すると行内で垂直線をまたぐ
        # 時，行の下端との交点を含む列と線分の通る列は異なる．
        # この場合は線分の通る列を下端点の属する列とする
        if len(rows_nox_V) > 0 and rows_nox_V[0] == row0:
            cols[0] = col0
        integral_nox_V = f[rows_nox_V, cols].sum()

        # (4-3)端の行で足しすぎた分
        extraintegral = (y0-row0)*f[row0, col0]+(row1+1-y1)*f[row1, col1]

        return (integral_x_V+integral_nox_V-extraintegral)/(y1-y0)*np.linalg.norm(end-start)

def graph_integral2d_0(f, G, pos):
    '''
    ２次元標本点上のスカラー場を無向グラフに沿って線積分する関数
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos: np.ndarray((V, 2)) Gの頂点の座標
    @return 線積分 \int_E(G) f dl
    '''
    retval = 0
    for u, v in G.edges_iter():
        retval += line_integral2d_0(f, pos[u], pos[v])

    return retval

def line_integral2d_grad_0(f, start, end):
    '''
    ２次元標本点上のスカラー場の線分に沿った線積分の，端点の座標に関する勾配ベクトル
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param start: np.ndarray(2) 始点
    @param end: np.ndarray(2) 終点
    @return 線積分の勾配ベクトルの組 [\grad_start\int f dl, \grad_end\int f dl]
    '''
    # 準備

    x0, y0 = start.tolist() # 始点
    x1, y1 = end.tolist() # 終点
    col0, row0 = start.astype(int).tolist() # 始点の属するセル
    col1, row1 = end.astype(int).tolist() # 終点の属するセル
    L = np.linalg.norm(end-start) # 線分の長さ

    # 例えば，始点のx座標による微分は以下のように書ける
    # d/dx0 \int_{P0->P1} f dl = (x0 - x1)/l^2*(\int f dl) + l*(\int_0^1 (1-t)*(df/dx(x(t), y(t))) dt)

    # 線積分の項

    integral_term = np.zeros((2, 2)) # [[x_i, y_i], [x_j, y_j]]
    integral_term[0] = np.array([x0-x1, y0-y1])/L**2*line_integral2d_0(f, start, end)
    integral_term[1] = -integral_term[0]

    # 偏微分の項

    pdif_term = np.zeros((2, 2)) # [[x_i, y_i], [x_j, y_j]]

    # x0, x1に関する微分．垂直線をまたぐ時の増分を調べる
    if x0 == x1:
        # y軸に平行な場合．垂直線をまたがないので0
        pdif_term[0][0] = 0
        pdif_term[1][0] = 0
    else:
        # y軸に平行でない場合
        dydx = (y1-y0)/(x1-x0) # 傾き
        yic = y0-dydx*x0 # y切片

        colL, colR = sorted([col0, col1]) # 最左列，最右列
        rows_x_V = (dydx*np.arange(colL+1, colR+1)+yic).astype(int) # 垂直線との交点を含む行のインデックス
        Df_x = f[rows_x_V, np.arange(colL+1, colR+1)] - f[rows_x_V, np.arange(colL, colR)] # 垂直線を左からまたぐ時の増分
        Df_x_sum = Df_x.sum()
        Df_x_dot_x = Df_x.dot(np.arange(colL+1, colR+1))

        if x0 < x1:
            # 始点が左の場合
            pdif_term[0][0] = (x1*Df_x_sum-Df_x_dot_x)/L
            pdif_term[1][0] = -(x0*Df_x_sum-Df_x_dot_x)/L
        else:
            # 始点が右の場合
            pdif_term[0][0] = -(x1*Df_x_sum-Df_x_dot_x)/L
            pdif_term[1][0] = (x0*Df_x_sum-Df_x_dot_x)/L

    # y0, y1に関する微分．水平線をまたぐ時の増分を調べる
    if y0 == y1:
        # x軸に平行な場合．水平線をまたがないので0
        pdif_term[0][1] = 0
        pdif_term[1][1] = 0
    else:
        # x軸に平行でない場合
        dxdy = (x1-x0)/(y1-y0) # 傾き
        xic = x0-dxdy*y0 # x切片

        rowB, rowT = sorted([row0, row1]) # 最下行，最上行
        cols_x_H = (dxdy*np.arange(rowB+1, rowT+1)+xic).astype(int) # 水平線との交点を含む列のインデックス
        Df_y = f[np.arange(rowB+1, rowT+1), cols_x_H] - f[np.arange(rowB, rowT), cols_x_H] # 水平線を下からまたぐ時の増分
        Df_y_sum = Df_y.sum()
        Df_y_dot_y = Df_y.dot(np.arange(rowB+1, rowT+1))

        if y0 < y1:
            # 始点が下の場合
            pdif_term[0][1] = (y1*Df_y_sum-Df_y_dot_y)/L
            pdif_term[1][1] = -(y0*Df_y_sum-Df_y_dot_y)/L
        else:
            # 始点が上の場合
            pdif_term[0][1] = -(y1*Df_y_sum-Df_y_dot_y)/L
            pdif_term[1][1] = (y0*Df_y_sum-Df_y_dot_y)/L

    return integral_term + pdif_term

def graph_integral2d_grad_0(f, G, pos):
    '''
    ２次元標本点上のスカラー場の無向グラフに沿った線積分の，頂点の座標に関する勾配ベクトル
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos: np.ndarray((V, 2)) Gの頂点の座標
    @return 線積分の勾配ベクトルのnp.array np.array([\grad_P\int_E(G) f dl for P \in V])
    '''
    V = len(G.nodes())
    retval = np.zeros((V, 2))

    for u, v in G.edges_iter():
        grad = line_integral2d_grad_0(f, pos[u], pos[v])
        retval[u] += grad[0]
        retval[v] += grad[1]

    return retval

def graphit_0(f, G, pos0, fixed=None, method='L-BFGS-B', constraints=[]):
    '''
    ２次元標本点上のスカラー場に，無向グラフの埋め込みをフィッティングする関数．
    最適化問題
    min \int_E(G) f dl s.t. (x_i, y_i) \in [0, M)x[0, N)
    を解く
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos0: np.ndarray((V, 2)) Gの頂点の座標の初期値
    @param fixed: np.ndarray((V, 2), dtype=bool) Gの頂点の座標を固定するかどうか
    @param method 最適化手法 'L-BFGS-B', 'SLSQP'
    @param constraints 制約辞書のリスト
    @return 最終的な頂点の座標とscipy.optimize.OptimizeResultオブジェクトの組

    制約辞書の例
    - 楕円上制約
    {
        'type': 'on_ellipse',
        'center': [center position],
        'a': [param a],
        'b': [param b],
        'vertex': [vertexes on ellipse]
    }
    '''
    if fixed is None:
        fixed = np.full_like(pos0, False, dtype=bool)
    if method == 'L-BFGS-B' and len(constraints) > 0:
        print('ERROR: L-BFGS-B doesn\'t support constraints.', file=sys.stderr)
        return None

    Rows, Cols = f.shape
    V = len(G.nodes())

    # 目的関数
    def func(x, sign=1):
        pos = np.zeros_like(pos0)
        pos[fixed] = pos0[fixed] # 固定された座標
        pos[np.logical_not(fixed)] = x # 固定されていない座標
        return sign*graph_integral2d_0(f, G, pos)

    # 勾配ベクトル
    def func_grad(x, sign=1):
        pos = np.zeros_like(pos0)
        pos[fixed] = pos0[fixed] # 固定された座標
        pos[np.logical_not(fixed)] = x # 固定されていない座標
        return sign*graph_integral2d_grad_0(f, G, pos)[np.logical_not(fixed)]

    # 制約条件 Vx(x, y)x(min, max)
    eps = 1e-6
    bounds = np.zeros((V, 2, 2))
    bounds[:, :, 0] = 0
    bounds[:, 0, 1] = Cols-eps
    bounds[:, 1, 1] = Rows-eps

    # その他の制約条件
    cons = []
    for constraint in constraints:
        if constraint['type'] == 'on_ellipse':
            # 楕円上制約の場合
            center = constraint['center']
            a = constraint['a']
            b = constraint['b']
            posid_to_xid = np.cumsum(np.logical_not(fixed.ravel()))-1
            for v in constraint['vertex']:
                con = {'type': 'eq'}
                xid = posid_to_xid[2*v]
                yid = posid_to_xid[2*v+1]
                con['fun'] = lambda x, sign=1: sign*((x[xid]-center[0])**2/a**2+(x[yid]-center[1])**2/b**2-1)
                def cons_grad(x, sign=1):
                    retval = np.zeros_like(x)
                    retval[xid] = 2*(x[xid]-center[0])/a**2
                    retval[yid] = 2*(x[yid]-center[1])/b**2
                    return sign*retval
                con['jac'] = cons_grad
                cons.append(con)

    if method == 'L-BFGS-B':
        res = opt.minimize(func, pos0[np.logical_not(fixed)], jac=func_grad, bounds=bounds[np.logical_not(fixed)], method=method, options={'disp': True})
    else:
        res = opt.minimize(func, pos0[np.logical_not(fixed)], jac=func_grad, bounds=bounds[np.logical_not(fixed)], constraints=cons, method=method, options={'disp': True})

    pos = np.zeros_like(pos0)
    pos[fixed] = pos0[fixed] # 固定された座標
    pos[np.logical_not(fixed)] = res.x # 固定されていない座標

    return pos, res

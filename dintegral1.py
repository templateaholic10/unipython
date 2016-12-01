# -*- coding: utf-8 -*-

'''
標本点上の関数の積分を提供するモジュール．
関数は標本点の1次スプラインによって矩形上に拡張する．
'''

import numpy as np
import networkx as nx
import scipy.optimize as opt

def _cell_H_coef(t1, t2):
    '''
    セルを横断する線積分の係数
    @param t1 線分と左辺の交点のy座標
    @param t2 線分と右辺の交点のy座標
    @return [[f_11, f_21], [f_12, f_22]]の係数
    '''
    return np.sqrt(1+(t2-t2)**2)*np.array((((1-t1)/3+(1-t2)/6, (1-t1)/6+(1-t2)/3), (t1/3+t2/6, t1/6+t2/3)))

def _cell_Q_coef(s, t):
    '''
    セルの右下をかすめる線積分の係数
    @param s 線分と下辺の交点のx座標
    @param t 線分と右辺の交点のy座標
    @return [[f_11, f_21], [f_12, f_22]]の係数
    '''
    return np.sqrt((1-s)**2+t**2)*np.array((((1-s)*(1-t)/6+(1-s)/3, s*(1-t)/6+s/3+(1-t)/3+1/6), ((1-s)*t/6, s*t/6+t/3)))

def _cell_3_coef(s, t1, t2):
    '''
    セルの右側に刺さった線積分の係数
    @param s 端点のx座標
    @param t1 端点のy座標
    @param t2 線分と右辺の交点のy座標
    @return [[f_11, f_21], [f_12, f_22]]の係数
    '''
    return np.sqrt((1-s)**2+(t2-t1)**2)*np.array((((1-s)*((1-t1)/3+(1-t2)/6), ((1-t1)/6+(1-t2)/3)+s*((1-t1)/3+(1-t2)/6)), ((1-s)*(t1/3+t2/6), (t1/6+t2/3)+s*(t1/3+t2/6))))

def line_integral2d(f, start, end):
    '''
    ２次元標本点上のスカラー場を線分に沿って線積分する関数
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M-1)x[0, N-1)上のスカラー場
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
    dx0, dy0 = x0-col0, y0-row0 # 始点の格子点からの変位
    dx1, dy1 = x1-col1, y1-row1 # 終点の格子点からの変位

    # ４つの場合に分けて計算する

    if y0 == y1:
        # (1)x軸に平行な線分．両端を1セルずつはみ出して積分して，端点の外側を引く
        integral = (1-dy0)*f[row0, col0:col1+2].sum() + dy0*f[row0+1, col0:col1+2].sum()
        integral_extra_L = (1-(1-dx0)**2/2)*((1-dy0)*f[row0, col0] + dy0*f[row0+1, col0]) + dx0**2/2*((1-dy0)*f[row0, col0+1] + dy0*f[row0+1, col0+1])
        integral_extra_R = (1-dx1**2/2)*((1-dy0)*f[row0, col1+1] + dy0*f[row0+1, col1+1]) + (1-dx1)**2/2*((1-dy0)*f[row0, col1] + dy0*f[row0+1, col1])
        return integral - integral_extra_L - integral_extra_R

    elif x0 == x1:
        # (2)y軸に平行な線分．両端を1セルずつはみ出して積分して，端点の外側を引く
        integral = (1-s0)*f[row0:row1+2, col0].sum() + s0*f[row0:row1+2, col0+1].sum()
        integral_extra_B = (1-(1-dy0)**2/2)*((1-dx0)*f[row0, col0] + dx0*f[row0, col0+1]) + dy0**2/2*((1-dx0)*f[row0+1, col0] + dx0*f[row0+1, col0+1])
        integral_extra_T = (1-dy1**2/2)*((1-dx0)*f[row1+1, col0] + dx0*f[row1+1, col0+1]) + (1-dy1)**2/2*((1-dx0)*f[row1, col0] + dx0*f[row1, col0+1])
        return integral - integral_extra_B - integral_extra_T

    elif abs(x1-x0) >= abs(y1-y0):
        # (3)寝た線分
        dydx = (y1-y0)/(x1-x0) # 傾きdy/dx
        yic = y0-dydx*x0 # y切片
        l = np.sqrt(1+dydx**2) # 単位長さ

        rowB, rowT = sorted([row0, row1]) # 最下行と最上行

        Hs = np.arange(rowB+1, rowT+1) # 水平線
        xs_x_H = (Hs-yic)/dydx # 水平線との交点のx座標
        cols_x_H = xs_x_H.astype(int) # 水平線との交点を含む列のインデックス
        cols_nox_H = np.setdiff1d(np.arange(col0, col1+1), cols_x_H) # 水平線との交点を含まない列のインデックス

        # (a) 水平線との交点を含まないセル

        y1s_nox_H = dydx*cols_nox_H+yic # 「水平線との交点を含まない列の左辺」との交点のy座標
        y2s_nox_H = dydx*(cols_nox_H+1)+yic # 「水平線との交点を含まない列の右辺」との交点のy座標
        rows_nox_H = y1s_nox_H.astype(int) # 水平線との交点を含まない列のセルの行
        if len(cols_nox_H) > 0 and cols_nox_H[0] == col0:
            # 水平線との交点を含まない列が左端の時，延長線上で水平線をまたぐ場合があるので，補正
            rows_nox_H[0] = row0
        dy1s_nox_H = y1s_nox_H-rows_nox_H # 「水平線との交点を含まない列の左辺」との交点のy変位
        dy2s_nox_H = y2s_nox_H-rows_nox_H # 「水平線との交点を含まない列の右辺」との交点のy変位

        cells_nox_H = l*(((1-dy1s_nox_H)/3+(1-dy2s_nox_H)/6)*f[rows_nox_H, cols_nox_H] + (dy1s_nox_H/3+dy2s_nox_H/6)*f[rows_nox_H+1, cols_nox_H] + ((1-dy1s_nox_H)/6+(1-dy2s_nox_H)/3)*f[rows_nox_H, cols_nox_H+1] + (dy1s_nox_H/6+dy2s_nox_H/3)*f[rows_nox_H+1, cols_nox_H]) # 水平線との交点を含まないセルの線積分
        integral_nox_H = cells_nox_H.sum()

        # (b) 水平線との交点を含む上下のセル

        dxs_x_H = xs_x_H-cols_x_H # 水平線との交点のx変位
        y1s_x_H = dydx*cols_x_H+yic # 「水平線との交点を含む列の左辺」との交点のy座標
        y2s_x_H = dydx*(cols_x_H+1)+yic # 「水平線との交点を含む列の右辺」との交点のy座標
        dy1s_x_H = y1s_x_H-Hs # 「水平線との交点を含む列の左辺」との交点のy変位
        dy2s_x_H = y2s_x_H-Hs # 「水平線との交点を含む列の右辺」との交点のy変位

        if dydx > 0:
            # 傾き正の場合
            cell1s_x_H = l*dxs_x_H*(((1-dxs_x_H)*(-dy1s_x_H)/6+(-dy1s_x_H)/3)*f[Hs-1, cols_x_H] + ((1-dxs_x_H)*(1+dy1s_x_H)/6+(1-dxs_x_H)/3+(1+dy1s_x_H)/3+1/6)*f[Hs, cols_x_H] + dxs_x_H*(-dy1s_x_H)/6*f[Hs-1, cols_x_H+1] + (dxs_x_H*(1+dy1s_x_H)/6+dxs_x_H/3)*f[Hs, cols_x_H+1]) # 水平線との交点を含む下のセルの線積分
            cell2s_x_H = l*(1-dxs_x_H)*(((1-dxs_x_H)*(1-dy2s_x_H)/6+(1-dxs_x_H)/3)*f[Hs, cols_x_H] + (1-dxs_x_H)*dy2s_x_H/6*f[Hs+1, cols_x_H] + (dxs_x_H*(1-dy2s_x_H)/6+dxs_x_H/3+(1-dy2s_x_H)/3+1/6)*f[Hs, cols_x_H+1] + (dxs_x_H*dy2s_x_H/6+dy2s_x_H/3)*f[Hs+1, cols_x_H+1]) # 水平線との交点を含む上のセルの線積分
            integral_x_H = cell1s_x_H.sum()+cell2s_x_H.sum()
        else:
            # 傾き負の場合
            cell1s_x_H = l*(1-dxs_x_H)*((1-dxs_x_H)*(-dy2s_x_H)/6*f[Hs-1, cols_x_H] + ((1-dxs_x_H)*(1+dy2s_x_H)/6+(1-dxs_x_H)/3)*f[Hs, cols_x_H] + (dxs_x_H*(-dy2s_x_H)/6+(-dy2s_x_H)/3)*f[Hs-1, cols_x_H+1] + (dxs_x_H*(1+dy2s_x_H)/6+dxs_x_H/3+(1+dy2s_x_H)/3+1/6)*f[Hs, cols_x_H+1]) # 水平線との交点を含む下のセルの線積分
            cell2s_x_H = l*dxs_x_H*(((1-dxs_x_H)*(1-dy1s_x_H)/6+(1-dxs_x_H)/3+(1-dy1s_x_H)/3+1/6)*f[Hs, cols_x_H] + ((1-dxs_x_H)*dy1s_x_H/6+dy1s_x_H/3)*f[Hs+1, cols_x_H] + (dxs_x_H*(1-dy1s_x_H)/6+dxs_x_H/3)*f[Hs, cols_x_H+1] + dxs_x_H*dy1s_x_H/6*f[Hs+1, cols_x_H+1]) # 水平線との交点を含む上のセルの線積分
            integral_x_H = cell1s_x_H.sum()+cell2s_x_H.sum()

        # (c) 端点の外側を引く

        y_L = dydx*col0+yic # 左端点の外側との交点のy座標
        dy_L = y_L-row0 # 左端点の外側との交点のy変位
        integral_extra_L = np.rot90(_cell_3_coef(1-dx0, 1-dy0, 1-dy_L), 2).ravel().dot(f[row0:row0+2, col0:col0+2].ravel())

        y_R = dydx*(col1+1)+yic # 右端点の外側との交点のy座標
        dy_R = y_R-row1 # 右端点の外側との交点のy変位
        integral_extra_R = _cell_3_coef(dx1, dy1, dy_R).ravel().dot(f[row1:row1+2, col1:col1+2].ravel())

        return integral_nox_H + integral_x_H - integral_extra_L - integral_extra_R

    else:
        # (4)立った線分
        dxdy = (x1-x0)/(y1-y0) # 傾きdx/dy
        xic = x0-dxdy*y0 # x切片
        l = np.sqrt(1+dxdy**2) # 単位長さ

        colL, colR = sorted([col0, col1]) # 最左列と最右列

        Vs = np.arange(colL+1, colR+1) # 垂直線
        ys_x_V = (Vs-xic)/dxdy # 垂直線との交点のy座標
        rows_x_V = ys_x_V.astype(int) # 垂直線と交点を含む行のインデックス
        rows_nox_V = np.setdiff1d(np.arange(row0, row1+1), rows_x_V) # 垂直線との交点を含まない行のインデックス

        # (a) 垂直線との交点を含まないセル

        x1s_nox_V = dxdy*rows_nox_V+xic # 「垂直線との交点を含まない行の下辺」との交点のx座標
        x2s_nox_V = dxdy*(rows_nox_V+1)+xic # 「垂直線との交点を含まない行の上辺」との交点のx座標
        cols_nox_V = x1s_nox_V.astype(int) # 垂直線との交点を含まない行のセルの列
        if len(rows_nox_V) > 0 and rows_nox_V[0] == row0:
            # 垂直線との交点を含まない行が下端の時，延長線上で垂直線をまたぐ場合があるので，補正
            cols_nox_V[0] = col0
        dx1s_nox_V = x1s_nox_V-cols_nox_V # 「垂直線との交点を含まない行の下辺」との交点のx変位
        dx2s_nox_V = x2s_nox_V-cols_nox_V # 「垂直線との交点を含まない行の上辺」との交点のx変位

        cells_nox_V = l*(((1-dx1s_nox_V)/3+(1-dx2s_nox_V)/6)*f[rows_nox_V, cols_nox_V] + ((1-dx1s_nox_V)/6+(1-dx2s_nox_V)/3)*f[rows_nox_V+1, cols_nox_V] + (dx1s_nox_V/3+dx2s_nox_V/6)*f[rows_nox_V, cols_nox_V+1] + (dx1s_nox_V/6+dx2s_nox_V/3)*f[rows_nox_V+1, cols_nox_V]) # 垂直線との交点を含まないセルの線積分
        integral_nox_V = cells_nox_V.sum()

        # (b) 垂直線との交点を含む左右のセル

        dys_x_V = ys_x_V-rows_x_V # 垂直線との交点のy変位
        x1s_x_V = dxdy*rows_x_V+xic # 「垂直線との交点を含む行の下辺」との交点のx座標
        x2s_x_V = dxdy*(rows_x_V+1)+xic # 「垂直線との交点を含む行の上辺」との交点のx座標
        dx1s_x_V = x1s_x_V-Vs # 「垂直線との交点を含む行の下辺」との交点のx変位
        dx2s_x_V = x2s_x_V-Vs # 「垂直線との交点を含む行の上辺」との交点のx変位

        if dxdy > 0:
            # 傾き正の場合
            cell1s_x_V = l*dys_x_V*(((-dx1s_x_V)*(1-dys_x_V)/6+(-dx1s_x_V)/3)*f[rows_x_V, Vs-1] + (-dx1s_x_V)*dys_x_V/6*f[rows_x_V+1, Vs-1] + ((1+dx1s_x_V)*(1-dys_x_V)/6+(1+dx1s_x_V)/3+(1-dys_x_V)/3+1/6)*f[rows_x_V, Vs] + ((1+dx1s_x_V)*dys_x_V/6+dys_x_V/3)*f[rows_x_V+1, Vs]) # 垂直線との交点を含む左のセルの線積分
            cell2s_x_V = l*(1-dys_x_V)*(((1-dx2s_x_V)*(1-dys_x_V)/6+(1-dys_x_V)/3)*f[rows_x_V, Vs] + ((1-dx2s_x_V)*dys_x_V/6+(1-dx2s_x_V)/3+dys_x_V/3+1/6)*f[rows_x_V+1, Vs] + dx2s_x_V*(1-dys_x_V)/6*f[rows_x_V, Vs+1] + (dx2s_x_V*dys_x_V/6+dx2s_x_V/3)*f[rows_x_V+1, Vs+1]) # 垂直線との交点を含む左のセルの線積分
            integral_x_V = cell1s_x_V.sum()+cell2s_x_V.sum()
        else:
            # 傾き負の場合
            cell1s_x_V = l*(1-dys_x_V)*((-dx2s_x_V)*(1-dys_x_V)/6*f[rows_x_V, Vs-1] + ((-dx2s_x_V)*dys_x_V/6+(-dx2s_x_V)/3)*f[rows_x_V+1, Vs-1] + ((1+dx2s_x_V)*(1-dys_x_V)/6+(1-dys_x_V)/3)*f[rows_x_V, Vs] + ((1+dx2s_x_V)*dys_x_V/6+(1+dx2s_x_V)/3+dys_x_V/3+1/6)*f[rows_x_V+1, Vs]) # 垂直線との交点を含む左のセルの線積分
            cell2s_x_V = l*dys_x_V*(((1-dx1s_x_V)*(1-dys_x_V)/6+(1-dx1s_x_V)/3+(1-dys_x_V)/3+1/6)*f[rows_x_V, Vs] + ((1-dx1s_x_V)*dys_x_V/6+dys_x_V/3)*f[rows_x_V+1, Vs] + (dx1s_x_V*(1-dys_x_V)/6+dx1s_x_V/3)*f[rows_x_V, Vs+1] + dx1s_x_V*dys_x_V/6*f[rows_x_V+1, Vs+1]) # 垂直線との交点を含む左のセルの線積分
            integral_x_V = cell1s_x_V.sum()+cell2s_x_V.sum()

        # (c) 端点の外側を引く

        x_B = dxdy*row0+xic # 下端点の外側との交点のx座標
        dx_B = x_B-col0 # 下端点の外側との交点のx変位
        integral_extra_B = np.rot90(_cell_3_coef(1-dy0, dx0, dx_B), 3).ravel().dot(f[row0:row0+2, col0:col0+2].ravel())

        x_T = dxdy*(row1+1)+xic # 上端点の外側との交点のx座標
        dx_T = x_T-col1 # 上端点の外側との交点のx変位
        integral_extra_T = np.rot90(_cell_3_coef(dy1, 1-dx1, 1-dx_T), 1).ravel().dot(f[row1:row1+2, col1:col1+2].ravel())

        return integral_nox_V + integral_x_V - integral_extra_B - integral_extra_T

def graph_integral2d(f, G, pos):
    '''
    ２次元標本点上のスカラー場を無向グラフに沿って線積分する関数
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos: np.ndarray((V, 2)) Gの頂点の座標
    @return 線積分 \int_E(G) f dl
    '''
    retval = 0
    for edge in G.edges():
        u, v = edge
        retval += line_integral2d(f, pos[u], pos[v])

    return retval

def line_integral2d_grad(f, start, end):
    '''
    ２次元標本点上のスカラー場の線分に沿った線積分の，端点の座標に関する勾配ベクトル
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param start: np.ndarray(2) 始点
    @param end: np.ndarray(2) 終点
    @return 線積分の勾配ベクトルの組 \grad_start\int f dl, \grad_end\int f dl
    '''
    # 準備

    x0, y0 = start.tolist() # 始点
    x1, y1 = end.tolist() # 終点
    col0, row0 = start.astype(int).tolist() # 始点の属するセル
    col1, row1 = end.astype(int).tolist() # 終点の属するセル
    l = np.linalg.norm(end-start) # 線分の長さ

    # 例えば，始点のx座標による微分は以下のように書ける
    # d/dx0 \int_{P0->P1} f dl = (x0 - x1)/l^2*(\int f dl) + l*(\int_0^1 (1-t)*(df/dx(x(t), y(t))) dt)

    # 第１項

    term1_P0 = np.array([x0-x1, y0-y1])/l**2*line_integral2d(f, start, end)
    term1_P1 = -term1_P0

    # 第２項

    term2_P0 = np.zeros(2)
    term2_P1 = np.zeros(2)

    # x0, x1に関する微分．垂直線をまたぐ時の増分を調べる
    if x0 == x1:
        # y軸に平行な場合．垂直線をまたがないので0
        term2_P0[0] = 0
        term2_P1[0] = 0
    else:
        # y軸に平行でない場合
        grad = (y1-y0)/(x1-x0) # 傾き
        yic = y0-grad*x0 # y切片

        colL, colR = sorted([col0, col1]) # 最左列，最右列
        rows_crossV = (grad*np.arange(colL+1, colR+1)+yic).astype(int) # 垂直線との交点を含む行のインデックス
        Df_x = f[rows_crossV, np.arange(colL+1, colR+1)] - f[rows_crossV, np.arange(colL, colR)] # 垂直線を左からまたぐ時の増分
        Df_x_sum = Df_x.sum()
        Df_x_dot_x = Df_x.dot(np.arange(colL+1, colR+1))

        if x0 < x1:
            # 始点が左の場合
            term2_P0[0] = (x1*Df_x_sum-Df_x_dot_x)/l
            term2_P1[0] = -(x0*Df_x_sum-Df_x_dot_x)/l
        else:
            # 始点が右の場合
            term2_P0[0] = -(x1*Df_x_sum-Df_x_dot_x)/l
            term2_P1[0] = (x0*Df_x_sum-Df_x_dot_x)/l

    # y0, y1に関する微分．水平線をまたぐ時の増分を調べる
    if y0 == y1:
        # x軸に平行な場合．水平線をまたがないので0
        term2_P0[1] = 0
        term2_P1[1] = 0
    else:
        # x軸に平行でない場合
        grad = (x1-x0)/(y1-y0) # 傾き
        xic = x0-grad*y0 # x切片

        rowB, rowT = sorted([row0, row1]) # 最下行，最上行
        cols_crossH = (grad*np.arange(rowB+1, rowT+1)+xic).astype(int) # 水平線との交点を含む列のインデックス
        Df_y = f[np.arange(rowB+1, rowT+1), cols_crossH] - f[np.arange(rowB, rowT), cols_crossH] # 水平線を下からまたぐ時の増分
        Df_y_sum = Df_y.sum()
        Df_y_dot_y = Df_y.dot(np.arange(rowB+1, rowT+1))

        if y0 < y1:
            # 始点が下の場合
            term2_P0[1] = (y1*Df_y_sum-Df_y_dot_y)/l
            term2_P1[1] = -(y0*Df_y_sum-Df_y_dot_y)/l
        else:
            # 始点が上の場合
            term2_P0[1] = -(y1*Df_y_sum-Df_y_dot_y)/l
            term2_P1[1] = (y0*Df_y_sum-Df_y_dot_y)/l

    return term1_P0 + term2_P0, term1_P1 + term2_P1

def graph_integral2d_grad(f, G, pos):
    '''
    ２次元標本点上のスカラー場の無向グラフに沿った線積分の，頂点の座標に関する勾配ベクトル
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos: np.ndarray((V, 2)) Gの頂点の座標
    @return 線積分の勾配ベクトルのnp.array np.array([\grad_P\int_E(G) f dl for P \in V])
    '''
    V = len(G.nodes())
    retval = np.zeros((V, 2))

    for edge in G.edges():
        u, v = edge
        grad_u, grad_v = line_integral2d_grad(f, pos[u], pos[v])
        retval[u] += grad_u
        retval[v] += grad_v

    return retval

def graph_fit(f, G, pos0, fixed=None, constraints=[]):
    '''
    ２次元標本点上のスカラー場に，無向グラフの埋め込みをフィッティングする関数．
    最適化問題
    min \int_E(G) f dl s.t. (x_i, y_i) \in [0, M)x[0, N)
    を逐次最小二乗法(SLSQP)で解く
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param G: networkx.Graph 0-originの整数をノード名とする無向グラフ
    @param pos0: np.ndarray((V, 2)) Gの頂点の座標の初期値
    @param fixed: np.ndarray((V, 2), dtype=bool) Gの頂点の座標を固定するかどうか
    @param constraints: 制約辞書のリスト
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

    N, M = f.shape
    V = len(G.nodes())

    # 目的関数
    def func(x, sign=1):
        pos = np.zeros_like(pos0)
        pos[fixed] = pos0[fixed] # 固定された座標
        pos[np.logical_not(fixed)] = x # 固定されていない座標
        return sign*graph_integral2d(f, G, pos)

    #func = lambda x, sign=1.0: sign*graph_integral2d(f, G, x.reshape((V, 2)))
    # 勾配ベクトル
    def func_grad(x, sign=1):
        pos = np.zeros_like(pos0)
        pos[fixed] = pos0[fixed] # 固定された座標
        pos[np.logical_not(fixed)] = x # 固定されていない座標
        return sign*graph_integral2d_grad(f, G, pos)[np.logical_not(fixed)]

    #func_grad = lambda x, sign=1.0: sign*graph_integral2d_grad(f, G, x.reshape((V, 2))).reshape(V*2)
    # 制約条件 Vx(x, y)x(min, max)
    eps = 1e-6
    bounds = np.zeros((V, 2, 2))
    bounds[:, :, 0] = 0
    bounds[:, 0, 1] = M-eps
    bounds[:, 1, 1] = N-eps

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

    res = opt.minimize(func, pos0[np.logical_not(fixed)], args=(-1, ), jac=func_grad, bounds=bounds[np.logical_not(fixed)], constraints=cons, method='SLSQP', options={'disp': True})

    pos = np.zeros_like(pos0)
    pos[fixed] = pos0[fixed] # 固定された座標
    pos[np.logical_not(fixed)] = res.x # 固定されていない座標

    return pos, res

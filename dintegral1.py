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
        # (1) x軸に平行な線分．両端を1セルずつはみ出して積分して，端点の外側を引く
        integral = (1-dy0)*f[row0, col0:col1+2].sum() + dy0*f[row0+1, col0:col1+2].sum()
        integral_extra_L = (1-(1-dx0)**2/2)*((1-dy0)*f[row0, col0] + dy0*f[row0+1, col0]) + dx0**2/2*((1-dy0)*f[row0, col0+1] + dy0*f[row0+1, col0+1])
        integral_extra_R = (1-dx1**2/2)*((1-dy0)*f[row0, col1+1] + dy0*f[row0+1, col1+1]) + (1-dx1)**2/2*((1-dy0)*f[row0, col1] + dy0*f[row0+1, col1])
        return integral - integral_extra_L - integral_extra_R

    elif x0 == x1:
        # (2) y軸に平行な線分．両端を1セルずつはみ出して積分して，端点の外側を引く
        integral = (1-s0)*f[row0:row1+2, col0].sum() + s0*f[row0:row1+2, col0+1].sum()
        integral_extra_B = (1-(1-dy0)**2/2)*((1-dx0)*f[row0, col0] + dx0*f[row0, col0+1]) + dy0**2/2*((1-dx0)*f[row0+1, col0] + dx0*f[row0+1, col0+1])
        integral_extra_T = (1-dy1**2/2)*((1-dx0)*f[row1+1, col0] + dx0*f[row1+1, col0+1]) + (1-dy1)**2/2*((1-dx0)*f[row1, col0] + dx0*f[row1, col0+1])
        return integral - integral_extra_B - integral_extra_T

    elif abs(x1-x0) >= abs(y1-y0):
        # (3) 寝た線分
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
        # (4) 立った線分
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

def _cell_grad_integrals(f, rows, cols, theta1, theta2, dx1, dy1, dx2, dy2):
    '''
    各セルの4種の線積分の和を返す関数
    '''
    ls = np.sqrt((dx2-dx1)**2+(dy2-dy1)**2) # セグメントの長さ
    integrals = np.zeros((2, 2)) # 返却する行列

    # x0に関する微分
    cells_x0 = ls*((f[rows, cols+1]-f[rows, cols])*((1-dy1)*((1-theta1)/3+(1-theta2)/6)+(1-dy2)*((1-theta1)/6+(1-theta2)/3)) + (f[rows+1, cols+1]-f[rows+1, cols])*(dy1*((1-theta1)/3+(1-theta2)/6)+dy2*((1-theta1)/6+(1-theta2)/3)))
    integrals[0, 0] = cells_x0.sum()

    # y0に関する微分
    cells_y0 = ls*((f[rows+1, cols]-f[rows, cols])*((1-dx1)*((1-theta1s)/3+(1-theta2s)/6)+(1-dx2)*((1-theta1s)/6+(1-theta2s)/3)) + (f[rows+1, cols+1]-f[rows, cols+1])*(dx1*((1-theta1)/3+(1-theta2)/6)+dx2*((1-theta1)/6+(1-theta2)/3)))
    integrals[0, 1] = cells_y0.sum()

    # x1に関する微分
    cells_x1 = ls*((f[rows, cols+1]-f[rows, cols])*((1-dy1)*(theta1/3+theta2/6)+(1-dy2)*(theta1/6+theta2/3)) + (f[rows+1, cols+1]-f[rows+1, cols])*(dy1*(theta1/3+theta2/6)+dy2*(theta1/6+theta2/3)))
    integrals[1, 0] = cells_x1.sum()

    # y1に関する微分
    cells_y1 = ls*((f[rows+1, cols]-f[rows, cols])*((1-dx1)*(theta1)/3+theta2)/6)+(1-dx2)*(theta1)/6+theta2)/3)) + (f[rows+1, cols+1]-f[rows, cols+1])*(dx1*(theta1/3+theta2/6)+dx2*(theta1/6+theta2/3)))
    integrals[1, 1] = cells_y1.sum()

    return integrals

def line_integral2d_grad(f, start, end):
    '''
    ２次元標本点上のスカラー場の線分に沿った線積分の，端点の座標に関する勾配ベクトル
    @param f: np.ndarray((N, M)) ２次元標本点上のスカラー場＝矩形[0, M)x[0, N)上のスカラー場
    @param start: np.ndarray(2) 始点
    @param end: np.ndarray(2) 終点
    @return 線積分の勾配ベクトルの組 \grad_start\int f dl, \grad_end\int f dl
    '''
    L = np.linalg.norm(end-start) # 線分全体の長さ

    # 第一項

    term1st = np.zeros((2, 2))
    term1st[0] = (start-end)/L**2*line_integral2d(f, start, end)
    term1st[1] = -term1st[0]

    # 第二項．４つの場合に分けて計算する

    if start[1] == end[1]:
        # (1) x軸に平行な場合
        P_L, P_R = start, end if start[0] < end[0] else end, start
        xL, y = P_L.tolist() # 左端点＝始点
        xR, _ = P_R.tolist() # 右端点＝終点
        colL, row = P_L.astype(int).tolist() # 左端点の属するセル
        colR, _ = P_R.astype(int).tolist() # 右端点の属するセル
        dxL, dy = xL-colL, y-row # 左端点の垂直線からの変位
        dxR, _ = xR-colR, y-row # 右端点の垂直線からの変位
        thetas = (np.arange(colL, colR+2)-xL)/(xR-xL)

        integrals_cover = _cell_grad_integrals(f, np.full(colR-colL, row), np.arange(colL, colR+1), thetas[:-1], thetas[1:], 0, dy, 1, dy)
        integrals_extra_L = _cell_grad_integrals(f, np.asarray((row)), np.asarray((colL)), thetas[0], 0, 0, dy, dxL, dy)
        integrals_extra_R = _cell_grid_integrals(f, np.asarray((row)), np.asarray((colR)), 1, thetas[-1], dxR, dy, 1, dy)
        integrals = integrals_cover - integrals_extra_L - integrals_extra_R

        term2nd = np.zeros((2, 2))
        term2nd[0], term2nd[1] = integrals[0], integrals[1] if start[0] < end[0] else integrals[1], integrals[0]

    elif start[0] == end[0]:
        # (2) y軸に平行な場合
        P_B, P_T = start, end if start[1] < end[1] else end, start
        x, yB = P_B.tolist() # 下端点＝始点
        _, yT = P_T.tolist() # 上端点＝終点
        col, rowB = P_B.astype(int).tolist() # 下端点の属するセル
        _, rowT = P_T.astype(int).tolist() # 上端点の属するセル
        dx, dyB = x-col, yB-rowB # 下端点の水平線からの変位
        _, dyT = x-col, yT-rowT # 上端点の水平線からの変位
        thetas = (np.arange(rowB, rowT+2)-yB)/(yT-yB)

        integrals_cover = _cell_grad_integrals(f, np.arange(rowB, rowT+1), np.full(rowT-rowB, col), thetas[:-1], thetas[1:], dx, 0, dx, 1)
        integrals_extra_B = _cell_grad_integrals(f, np.asarray((rowB)), np.asarray((col)), thetas[0], 0, dx, 0, dx, dyB)
        integrals_extra_T = _cell_grid_integrals(f, np.asarray((rowT)), np.asarray((col)), 1, thetas[-1], dx, dyT, dx, 1)
        integrals = integrals_cover - integrals_extra_B - integrals_extra_T

        term2nd = np.zeros((2, 2))
        term2nd[0], term2nd[1] = integrals[0], integrals[1] if start[1] < end[1] else integrals[1], integrals[0]

    elif abs(end[0]-start[0]) >= abs(end[1]-start[1]):
        # (3) 寝た線分

        P_L, P_R = start, end if start[0] < end[0] else end, start
        xL, yL = P_L.tolist() # 左端点＝始点
        xR, yR = P_R.tolist() # 右端点＝終点
        colL, rowL = P_L.astype(int).tolist() # 左端点の属するセル
        colR, rowR = P_R.astype(int).tolist() # 右端点の属するセル
        dxL, dyL = xL-colL, y-row # 左端点の垂直線からの変位
        dxR, dyR = xR-colR, y-row # 右端点の垂直線からの変位

        dydx = (yR-yL)/(xR-xL) # 傾きdy/dx
        yic = yL-dydx*xL # y切片
        lx = np.sqrt(1+dydx**2) # x軸方向の単位長さ

        rowB, rowT = sorted([rowL, rowR]) # 最下行と最上行

        Hs = np.arange(rowB+1, rowT+1) # 水平線
        xs_x_H = (Hs-yic)/dydx # 水平線との交点のx座標
        cols_x_H = xs_x_H.astype(int) # 水平線との交点を含む列のインデックス
        cols_nox_H = np.setdiff1d(np.arange(colL, colR+1), cols_x_H) # 水平線との交点を含まない列のインデックス

        # (a) 水平線との交点を含まないセル

        theta1s_nox_H = (cols_nox_H-xL)/(xR-xL) # 「水平線との交点を含まない列の左辺」までの線分全体に占める割合
        theta2s_nox_H = (cols_nox_H+1-xL)/(xR-xL) # 「水平線との交点を含まない列の右辺」までの線分全体に占める割合
        dx1_nox_H, dx2_nox_H = 0, 1 # 水平線との交点を含まないセルとの交点のx変位
        y1s_nox_H = dydx*cols_nox_H+yic # 「水平線との交点を含まない列の左辺」との交点のy座標
        y2s_nox_H = dydx*(cols_nox_H+1)+yic # 「水平線との交点を含まない列の右辺」との交点のy座標
        rows_nox_H = y1s_nox_H.astype(int) # 水平線との交点を含まない列のセルの行
        if len(cols_nox_H) > 0 and cols_nox_H[0] == col0:
            # 水平線との交点を含まない列が左端の時，延長線上で水平線をまたぐ場合があるので，補正
            rows_nox_H[0] = row0
        dy1s_nox_H = y1s_nox_H-rows_nox_H # 「水平線との交点を含まない列の左辺」との交点のy変位
        dy2s_nox_H = y2s_nox_H-rows_nox_H # 「水平線との交点を含まない列の右辺」との交点のy変位

        integrals_nox_H = _cell_grad_integrals(f, rows_nox_H, cols_nox_H, theta1s_nox_H, theta2s_nox_H, dx1_nox_H, dy1s_nox_H, dx2_nox_H, dy2s_nox_H)

        # (b) 水平線との交点を含む上下のセル

        theta1s_x_H = (cols_x_H-xL)/(xR-xL) # 「水平線との交点を含む列の左辺」までの線分全体に占める割合
        theta2s_x_H = (cols_x_H+1-xL)/(xR-xL) # 「水平線との交点を含む列の右辺」までの線分全体に占める割合
        thetaMs_x_H = (Hs-yL)/(yR-yL) # 「水平線との交点」までの線分全体に占める割合
        dxMs_x_H = xs_x_H-cols_x_H # 水平線との交点のx変位
        y1s_x_H = dydx*cols_x_H+yic # 「水平線との交点を含む列の左辺」との交点のy座標
        y2s_x_H = dydx*(cols_x_H+1)+yic # 「水平線との交点を含む列の右辺」との交点のy座標
        dy1s_x_H = y1s_x_H-y1s_x_H_astype(int) # 「水平線との交点を含む列の左辺」との交点のy変位
        dy2s_x_H = y2s_x_H-y2s_x_H.astype(int) # 「水平線との交点を含む列の右辺」との交点のy変位

        if dydx > 0:
            # 傾き正の場合
            integrals_x_H_B = _cell_grad_integrals(f, Hs-1, cols_x_H, theta1s_x_H, thetaMs_x_H, 0, dy1s_x_H, dxMs_x_H, 1) # 下のセル

            integrals_x_H_T = _cell_grad_integrals(f, Hs, cols_x_H, thetaMs_x_H, theta2s_x_H, dxMs_x_H, 0, 1, dy2s_x_H) # 上のセル
        else:
            # 傾き負の場合
            integrals_x_H_B = _cell_grad_integrals(f, Hs-1, cols_x_H, thetaMs_x_H, theta2s_x_H, dxMs_x_H, 1, 1, dy2s_x_H) # 下のセル

            integrals_x_H_T = _cell_grad_integrals(f, Hs, cols_x_H, theta1s_x_H, thetaMs_x_H, 0, dy1s_x_H, dxMs_x_H, 0) # 上のセル

        # (c) 端点の外側を引く

        thetaLL = (colL-xL)/(xR-xL) # 左外側端点の線分に占める割合(<0)
        yLL = dydx*colL+yic # 左外側端点のy座標
        dyLL = yLL-rowL # 左外側端点のy変位．0-1に限らない
        integrals_extra_L = _cell_grad_integrals(f, np.asarray((rowL)), np.asarray((colL)), thetaLL, 0, 0, dyLL, dxL, dyL)

        thetaRR = (colR+1-xL)/(xR-xL) # 右外側端点の線分に占める割合(>1)
        yRR = dydx*(colR+1)+yic # 右外側端点のy座標
        dyRR = yRR-rowR # 右外側端点のy変位．0-1に限らない
        integrals_extra_R = _cell_grad_integrals(f, np.asarray((rowR)), np.asarray((colR)), 1, thetaRR, dxR, dyR, 1, dyRR)

        integrals = integrals_nox_H + integrals_x_H_B + integrals_x_H_T - integrals_extra_L - integrals_extra_R

        term2nd = np.zeros((2, 2))
        term2nd[0], term2nd[1] = integrals[0], integrals[1] if start[0] < end[0] else integrals[1], integrals[0]

    else:
        # (4) 立った線分
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

    return term1st + term2nd

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
        grad = line_integral2d_grad(f, pos[u], pos[v])
        retval[u] += grad[0]
        retval[v] += grad[1]

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

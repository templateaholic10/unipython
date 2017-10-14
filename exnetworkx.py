# -*- coding: utf-8 -*-

"""
# exnetworkxモジュール

networkxの拡張
"""

import matplotlib.pyplot as plt
import networkx as nx

def draw3d(G, pos, zorder, ax=None, node_params={}, edge_params={}, margin_width=None, bgcolor='w'):
    '''
    グラフの描画関数．辺や頂点の交差を表現する
    @param G: グラフ
    @param pos: 頂点座標(dict of G.nodes())
    @param zorder: 辺および頂点のzorder(dict of G.nodes()+G.edges())
    @param ax: Axesオブジェクト
    @param node_params: scatterのパラメータ．dict of G.nodes()を渡すと頂点ごとに異なるパラメータで描画できる
    @param edge_params: plotのパラメータ．dict of G.edges()を渡すと辺ごとに異なるパラメータで描画できる．
    @param margin_width: 手前の辺のマージン幅
    @return res: res['ax']はAxesオブジェクト，res['obj']はカラーバー用のscatterオブジェクト
    '''
    # パラメータの準備
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if 'c' not in node_params:
        node_params['c'] = 'k'
    if 'c' not in edge_params:
        edge_params['c'] = 'k'
    if 'lw' not in edge_params:
        edge_params['lw'] = 1.
    eps = 1e-6

    # 頂点の描画
    for v in G.nodes_iter():
        _node_params = {key: (node_params[key][v] if isinstance(node_params[key], dict) else node_params[key]) for key in node_params}
        obj = ax.scatter(pos[v][0], pos[v][1], zorder=zorder[v], **_node_params)

    # 辺の描画
    for u, v in G.edges_iter():
        _edge_params = {key: (edge_params[key][(u, v)] if isinstance(edge_params[key], dict) else edge_params[key]) for key in edge_params}
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], zorder=zorder[(u, v)], **_edge_params)
        _margin_width = _edge_params['lw'] if margin_width is None else margin_width
        ax.plot([pos[u][0], pos[v][0]], [pos[u][1], pos[v][1]], zorder=zorder[(u, v)]-eps, lw=_edge_params['lw']+2.*_margin_width, c=bgcolor)

    res = {'ax': ax, 'obj': obj}
    return res

# -*- coding: utf-8 -*-

"""
# gspplotモジュール

グラフ信号を描画する関数群を収めたモジュール．
グラフはnetworkx.Graphで表現する．
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

def colorplot(G, f, ax=None, pos=None, with_labels=False, colorbar=True, **args):
    """
    グラフ信号を頂点の色で表す描画関数
    @param G networkx.Graph
    @param f 信号ベクトル
    @param ax axes．なければ作成
    @param pos グラフのレイアウト．なければspring_layoutで作成
    @param colorbar カラーバーの表示非表示．デフォルトは表示
    @param **args draw_networkx_nodes，draw_networkx_edgesの引数
    @return axes
    """
    if ax is None:
        ax = plt.gca()
    if pos is None:
        pos = nx.spring_layout(G)
    if 'edge_color' not in args:
        args['edge_color'] = colors.cnames['blue']
    # グラフの描画
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, cmap=cm.gnuplot, node_color=f, **args)
    if with_labels:
        nx.draw_networkx_labels(G, pos, ax=ax)
    nx.draw_networkx_edges(G, pos, ax=ax, **args)
    # カラーバーの描画
    if colorbar:
        plt.colorbar(nodes, ax=ax)
    ax.axis('off')
    return ax

def barplot(G, f, ax=None, pos=None, with_labels=False, edge_color=None, bar_color=None, **args):
    """
    グラフ信号を頂点上のバーで表す描画関数
    @param G networkx.Graph
    @param f 信号ベクトル
    @param ax axes3d．なければ作成
    @param pos グラフのレイアウト．なければspring_layoutで作成
    @param edge_color 辺の色
    @param bar_color バーの色
    @param **args plotの引数
    @return axes
    """
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if pos is None:
        pos = nx.spring_layout(G)
    if edge_color is None:
        edge_color = colors.cnames['blue']
    if bar_color is None:
        bar_color = colors.cnames['red']
    # グラフの描画
    for edge in G.edges():
        u, v = edge
        ax.plot((pos[u][0], pos[v][0]), (pos[u][1], pos[v][1]), zs=0, color=edge_color, **args)
    # バーの描画
    for i, node in enumerate(sorted(G.nodes())):
        ax.plot([pos[node][0]]*2, [pos[node][1]]*2, zs=[0, f[i]], color=bar_color, **args)
        if with_labels:
            ax.text(pos[node][0], pos[node][1], 0, node)
    ax.set_axis_off()
    return ax

def spectrumplot(l, F, ax=None, **args):
    """
    グラフ信号のスペクトルの描画関数
    @param l 周波数のリスト
    @param F スペクトルベクトル
    @param ax axes．なければ作成
    @param **args plotの引数
    @return axes
    """
    if ax is None:
        ax = plt.subplot(111)
    # 描画
    ax.plot(l, F, **args)
    ax.set_xlim(min(l), max(l))
    return ax

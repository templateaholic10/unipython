# -*- coding: utf-8 -*-

"""
# gspplot$B%b%8%e!<%k(B

$B%0%i%U?.9f$rIA2h$9$k4X?t72$r<}$a$?%b%8%e!<%k!%(B
$B%0%i%U$O(Bnetworkx.Graph$B$GI=8=$9$k!%(B
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
import networkx as nx
from mpl_toolkits.mplot3d import Axes3D

_colormap = cm.gnuplot
_edge_color = colors.cnames['blue']
_bar_color = colors.cnames['red']

def colorplot(G, f, ax=None, pos=None, colorbar=True, **args):
    """
    $B%0%i%U?.9f$rD:E@$N?'$GI=$9IA2h4X?t(B
    @param G networkx.Graph
    @param f $B?.9f%Y%/%H%k(B
    @param ax axes$B!%$J$1$l$P:n@.(B
    @param pos $B%0%i%U$N%l%$%"%&%H!%$J$1$l$P(Bspring_layout$B$G:n@.(B
    @param colorbar $B%+%i!<%P!<$NI=<(HsI=<(!%%G%U%)%k%H$OI=<((B
    @param **args draw_networkx_nodes$B!$(Bdraw_networkx_edges$B$N0z?t(B
    @return axes
    """
    if ax is None:
        ax = plt.subplot(111)
    if pos is None:
        pos = nx.spring_layout(G)

    # $B%0%i%U$NIA2h(B
    nodes = nx.draw_networkx_nodes(G, pos, ax=ax, cmap=_colormap, node_color=f, **args)
    if 'edge_color' in args:
        nx.draw_networkx_edges(G, pos, ax=ax, **args)
    else:
        nx.draw_networkx_edges(G, pos, ax=ax, edge_color=_edge_color, **args)

    # $B%+%i!<%P!<$NIA2h(B
    if colorbar:
        plt.colorbar(nodes, ax=ax)

    ax.axis('off')

    return ax

def barplot(G, f, ax=None, pos=None, **args):
    """
    $B%0%i%U?.9f$rD:E@>e$N%P!<$GI=$9IA2h4X?t(B
    @param G networkx.Graph
    @param f $B?.9f%Y%/%H%k(B
    @param ax axes3d$B!%$J$1$l$P:n@.(B
    @param pos $B%0%i%U$N%l%$%"%&%H!%$J$1$l$P(Bspring_layout$B$G:n@.(B
    @param **args plot$B$N0z?t(B
    @return axes
    """
    if ax is None:
        ax = plt.subplot(111, projection='3d')
    if pos is None:
        pos = nx.spring_layout(G)

    # $B%0%i%U$NIA2h(B
    for edge in G.edges():
        u, v = edge
        if 'color' in args:
            ax.plot((pos[u][0], pos[v][0]), (pos[u][1], pos[v][1]), zs=0, **args)
        else:
            ax.plot((pos[u][0], pos[v][0]), (pos[u][1], pos[v][1]), zs=0, color=_edge_color, **args)

    # $B%P!<$NIA2h(B
    for i, node in enumerate(G.nodes()):
        if 'color' in args:
            ax.plot([pos[node][0]]*2, [pos[node][1]]*2, zs=[0, f[i]], **args)
        else:
            ax.plot([pos[node][0]]*2, [pos[node][1]]*2, zs=[0, f[i]], color=_bar_color, **args)
    
    ax.set_axis_off()
    
    return ax

def spectrumplot(l, F, ax=None, **args):
    """
    $B%0%i%U?.9f$N%9%Z%/%H%k$NIA2h4X?t(B
    @param l $B<~GH?t$N%j%9%H(B
    @param F $B%9%Z%/%H%k%Y%/%H%k(B
    @param ax axes$B!%$J$1$l$P:n@.(B
    @param **args plot$B$N0z?t(B
    @return axes
    """
    if ax is None:
        ax = plt.subplot(111)

    # $BIA2h(B
    ax.plot(l, F, **args)

    ax.set_xlim(min(l), max(l))

    return ax
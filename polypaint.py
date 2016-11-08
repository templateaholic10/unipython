# -*- coding: utf-8 -*-

'''
対話的にグラフ配置を作成するクラスを提供するモジュール．
スクリプトとして実行することもできる
'''

import numpy as np
import networkx as nx
import cv2
import copy
import sys
import os
import pickle

class Canvas(object):
    '''
    対話的なグラフ配置環境クラス
    '''
    __MAXSIZE = (1000, 700) # 最大背景サイズ
    __NBHD = 10 # 近傍の半径

    radius = 5 # 頂点の半径
    vertex_color = (54, 67, 244) # 頂点の色
    select_radius = 8 # 選択中の頂点の半径
    select_vertex_color = (54, 67, 244) # 選択中の頂点の色
    hover_radius = 8 # ホバー中の頂点の半径
    hover_vertex_color = (7, 192, 254) # ホバー中の頂点の色
    hold_radius = 8 # ホールド中の頂点の半径
    hold_vertex_color = (7, 192, 254) # ホールド中の頂点の色
    width = 3 # 辺の幅
    edge_color = (0, 0, 0) # 辺の色

    def __init__(self, filename=None, size=None, G=None, pos=None):
        '''
        コンストラクタ
        @param filename 背景画像．なければ白背景になる
        @param size 背景サイズ
        @param G 無向グラフ
        @param pos 頂点の座標
        '''
        # 背景画像の設定
        if filename is None:
            # 背景画像を指定しない時，白紙にする
            if size is None:
                size = Canvas.__MAXSIZE
            self.background = np.full((size[1], size[0], 3), 255, dtype=np.uint8)
            self.redratio = 1
        else:
            self.background = cv2.imread(filename, 1)
            size_org = (self.background.shape[1], self.background.shape[0])
            if size_org[0] > Canvas.__MAXSIZE[0] or size_org[1] > Canvas.__MAXSIZE[1]:
                # 縮小
                self.redratio = min(Canvas.__MAXSIZE[0]/size_org[0], Canvas.__MAXSIZE[1]/size_org[1]) # 縮小率
            else:
                self.redratio = 1
            self.background = cv2.resize(self.background, None, fx=self.redratio, fy=self.redratio)

        # グラフとその配置
        if G is None or pos is None:
            self.G = nx.Graph()
            self.pos =dict()
            self.nth_vertex = 0
        else:
            self.G = G
            self.pos = {v: (pos[i]*self.redratio).astype(int) for i, v in enumerate(self.G.nodes())}
            self.nth_vertex = len(self.G.nodes())

        # マウスイベント用の変数
        self.select = None
        self.hover = None
        self.hold = None

    def __mouse_event(self, event, x, y, flags, param):
        '''
        マウスイベント時の処理
        @param event イベントの種類
        @param x x座標
        @param y y座標
        @param flags イベントに付随するフラグ．ビットマスクで取り出す
        @param param その他のパラメータ
        '''
        P = np.array((x, y))

        if event == cv2.EVENT_LBUTTONDOWN:
            # (1)左ボタン押下した時
            # 点をホールドする
            print('left button down at {0}'.format(P))
            if self.hover is None:
                # (1-1)ホバー中でない時
                # 新しい点を作成してホールドする
                self.G.add_node(self.nth_vertex)
                self.pos[self.nth_vertex] = P
                self.hold = self.nth_vertex
                self.nth_vertex += 1
            else:
                # (1-2)ホバー中の時
                # ホバー中の点をホールドする
                self.hold = self.hover
                self.hover = None

        elif event == cv2.EVENT_RBUTTONDOWN:
            # (2)右ボタン押下した時
            # 点に選択中の点から線を引き，ホールドする
            print('right button down at {0}'.format(P))
            if self.select is None:
                # (2-1)選択中でない時
                # 何もしない
                pass
            elif self.hover is None:
                # (2-2)選択中だがホバー中でない時
                # 新しい点を作成して線を引き，ホールドする
                self.G.add_node(self.nth_vertex)
                self.pos[self.nth_vertex] = P
                self.hold = self.nth_vertex
                self.nth_vertex += 1
                self.G.add_edge(self.select, self.hold)
            else:
                # (2-3)選択中でホバー中の時
                # ホバー中の点へ線を引き，ホールドする
                self.hold = self.hover
                self.hover = None
                self.G.add_edge(self.select, self.hold)

        elif event == cv2.EVENT_LBUTTONUP or event == cv2.EVENT_RBUTTONUP:
            # (3)左ボタンまたは右ボタンを離した時
            # ホールド中の点を解除して選択する
            print('button up at {0}'.format(P))
            self.select = self.hold
            self.hold = None

        elif event == cv2.EVENT_MOUSEMOVE:
            # (4)マウスを動かした時
            print('mouse move to {0}'.format(P))

            if self.hold is not None:
                # ホールド中なら移動する
                self.pos[self.hold] = P
            elif self.pos:
                # ホールド中でなく，頂点が存在する時ホバー判定する
                v_closest, P_closest = min(self.pos.items(), key=lambda x: np.linalg.norm(P-x[1]))
                if np.linalg.norm(P-P_closest) < Canvas.__NBHD:
                    self.hover = v_closest
                else:
                    self.hover = None

    def run(self):
        '''
        描画を開始する関数
        @return グラフと座標辞書のタプル
        '''
        cv2.namedWindow('PolyPaint')
        cv2.setMouseCallback('PolyPaint', self.__mouse_event)

        frame = 0
        while True:
            print('frame: {0}, select: {1}, hover: {2}, hold: {3}'.format(frame, self.select, self.hover, self.hold))
            img = copy.deepcopy(self.background)

            # 辺の描画
            for edge in self.G.edges():
                P1, P2 = self.pos[edge[0]], self.pos[edge[1]]
                cv2.line(img, (P1[0], P1[1]), (P2[0], P2[1]), Canvas.edge_color, Canvas.width)

            # 頂点の描画
            for v in self.pos:
                if v == self.hover:
                    cv2.circle(img, (self.pos[v][0], self.pos[v][1]), Canvas.hover_radius, Canvas.hover_vertex_color, -1)
                elif v == self.hold:
                    cv2.circle(img, (self.pos[v][0], self.pos[v][1]), Canvas.hold_radius, Canvas.hold_vertex_color, -1)
                elif v == self.select:
                    cv2.circle(img, (self.pos[v][0], self.pos[v][1]), Canvas.select_radius, Canvas.select_vertex_color, -1)
                else:
                    cv2.circle(img, (self.pos[v][0], self.pos[v][1]), Canvas.radius, Canvas.vertex_color, -1)

            # 表示
            cv2.imshow('PolyPaint', img)

            key = cv2.waitKey(100) & 0xFF

            if key == 127:
                # deleteキーを押下した時，選択中の頂点を削除
                if self.select is not None:
                    self.G.remove_node(self.select)
                    self.pos.pop(self.select)
                    self.select = None
            elif key == ord('q'):
                # 'q'を押下した時，終了
                break

            frame += 1

        cv2.destroyAllWindows()

        # 頂点座標は原寸に戻してnp.arrayにする
        pos = np.array([self.pos[v] for v in self.G.nodes()])/self.redratio
        # グラフの頂点番号は詰める
        G = nx.convert_node_labels_to_integers(self.G)
        return G, pos

def main():
    '''
    モジュールを実行した時のエントリポイント．
    対話的にグラフ配置を作成し，networkx.Graphとposのタプルをpickleで保存する．

    python polypaint.py (background filename) (pickle filename)

    - background filename
        背景画像．指定しなければ白背景になる
    - pickle filename
        networkx.Graphとposのタプルのpickle．なければ新規作成する
    '''
    bg_filename = None if len(sys.argv) <= 1 else sys.argv[1]
    if len(sys.argv) <= 2:
        G0 = pos0 = None
    else:
        pickle_filename = sys.argv[2]
        with open(pickle_filename, 'rb') as fin:
            G0, pos0 = pickle.load(fin)

    canvas = Canvas(bg_filename, G=G0, pos=pos0)
    G, pos = canvas.run()

    if bg_filename is None:
        foutname = 'graph.pickle'
    else:
        foutname = 'graph_on_'+os.path.splitext(os.path.basename(bg_filename))[0]+'.pickle'

    with open(foutname, 'wb') as fout:
        pickle.dump((G, pos), fout)

if __name__ == '__main__':
    main()

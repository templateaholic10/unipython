# -*- coding: utf-8 -*-

'''
# PIPモジュール
確率的画像処理を提供するモジュール
'''

import numpy as np
import networkx as nx
import numbers
from scipy.sparse import lil_matrix
from sklearn.preprocessing import normalize

def compressed_adjacency_matrix(M, N):
    Ve, Vo = (M*N+1)//2, M*N//2 # 偶頂点と奇頂点の数
    A_eo = lil_matrix((Ve, Vo), dtype=int)
    
    ote = lambda ve: ve-(N+1)//2
    eto = lambda vo: vo-N//2
    ole = lambda ve: ve-int((N%2!=0)or(ve//(N//2))%2==0)
    elo = lambda vo: vo-int((N%2==0)and(vo//(N//2))%2==1)
    
    # 偶頂点を舐めて左または上の奇頂点に線を引く
    for ve in range(1, Ve):
        if ve < (N+1)//2:
            A_eo[ve, ole(ve)] = 1 # 上辺上ならば左のみ
        elif ve%N == 0:
            A_eo[ve, ote(ve)] = 1 # 左辺上ならば上のみ
        else:
            A_eo[ve, ole(ve)] = A_eo[ve, ote(ve)] = 1
    
    # 奇頂点を舐めて左または上の偶頂点に線を引く
    for vo in range(Vo):
        if vo < N//2:
            A_eo[elo(vo), vo] = 1 # 上辺上ならば左のみ
        elif vo%N == N//2:
            A_eo[eto(vo), vo] = 1 # 左辺上ならば上のみ
        else:
            A_eo[elo(vo), vo] = A_eo[eto(vo), vo] = 1
            
    return A_eo

class MRF(object):
    '''
    Markov Random Fieldクラス
    '''
    def __init__(self, size, L, beta=1.75):
        '''
        コンストラクタ
        @param size MRFの大きさ(M, N)
        @param L ラベル数
        @param beta ラベル間の結合度行列
        '''
        self.size = size
        self.L = L

        if isinstance(beta, numbers.Number):
            self.beta = beta*np.identity(L)
        else:
            self.beta = beta

    def pdf(self):
        pass

    def rvs(self, length=1, burn_in=100, X0=None):
        '''
        サンプリング関数
        @param length サンプル数
        @param burn_in 棄却期間
        @param X0 初期状態
        '''
        if X0 is None:
            X0 = np.random.choice(self.L, self.size)
        M, N = self.size
        Evens = np.asarray([[(m%2==0 and n%2==0) or (m%2==1 and n%2==1) for n in range(N)] for m in range(M)], dtype=bool).ravel() # 偶頂点のboolean index(M*N, )
        Odds = np.logical_not(Evens) # 奇頂点のboolean index(M*N, )

        A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(nx.grid_2d_graph(M, N), ordering='sorted')) # 隣接行列(M*N, M*N)
        A_eo = A[Evens][:, Odds] # 隣接行列の奇頂点->偶頂点部分
        A_oe = A[Odds][:, Evens] # 隣接行列の偶頂点->奇頂点部分

        X_e = X0.ravel()[Evens] # 現在のラベルの偶頂点部分
        X_o = X0.ravel()[Odds] # 現在のラベルの奇頂点部分

        for step in range(burn_in):
            # 偶頂点の更新

            # 隣接頂点のラベルから多項分布のパラメータを作成
            nbr_labels = np.asarray([A_eo.dot((X_o==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            thetas = np.exp(nbr_labels.dot(self.beta)) # 多項分布のパラメータ（非正規）(M*N/2, L)
            thetas /= thetas.sum(axis=1, keepdims=True) # 多項分布のパラメータ(M*N/2, L)

            # 多項分布からサンプリング
            X_e = np.apply_along_axis(lambda theta: np.random.choice(self.L, p=theta), 1, thetas)

            # 奇頂点の更新

            # 隣接頂点のラベルから多項分布のパラメータを作成
            nbr_labels = np.asarray([A_oe.dot((X_e==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            thetas = np.exp(nbr_labels.dot(self.beta)) # 多項分布のパラメータ（非正規）(M*N/2, L)
            thetas /= thetas.sum(axis=1, keepdims=True) # 多項分布のパラメータ(M*N/2, L)

            # 多項分布からサンプリング
            X_o = np.apply_along_axis(lambda theta: np.random.choice(self.L, p=theta), 1, thetas)

        retval = np.zeros((length, M*N)) # サンプル
        retval[0][Evens] = X_e
        retval[0][Odds] = X_o

        for step in range(1, length):
            # 偶頂点の更新

            # 隣接頂点のラベルから多項分布のパラメータを作成
            nbr_labels = np.asarray([A_eo.dot((X_o==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            thetas = np.exp(nbr_labels.dot(self.beta)) # 多項分布のパラメータ（非正規）(M*N/2, L)
            thetas /= thetas.sum(axis=1, keepdims=True) # 多項分布のパラメータ(M*N/2, L)

            # 多項分布からサンプリング
            X_e = np.apply_along_axis(lambda theta: np.random.choice(self.L, p=theta), 1, thetas)

            # 奇頂点の更新

            # 隣接頂点のラベルから多項分布のパラメータを作成
            nbr_labels = np.asarray([A_oe.dot((X_e==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            thetas = np.exp(nbr_labels.dot(self.beta)) # 多項分布のパラメータ（非正規）(M*N/2, L)
            thetas /= thetas.sum(axis=1, keepdims=True) # 多項分布のパラメータ(M*N/2, L)

            # 多項分布からサンプリング
            X_o = np.apply_along_axis(lambda theta: np.random.choice(self.L, p=theta), 1, thetas)

            retval[step][Evens] = X_e
            retval[step][Odds] = X_o

        return retval.reshape((length, *self.size))

class HMRF(object):
    '''
    Hidden Markov Random Fieldクラス．
    マルコフ確率場はdiscrete colors，観測はGaussian emission
    '''
    def __init__(self, size, L, mus, sigma2s=None, beta=1.75):
        '''
        コンストラクタ
        @param size MRFの大きさ(M, N)
        @param L ラベル数
        @param mus ラベルごとの観測平均
        @param sigma2s ラベルごとの観測分散
        @param beta ラベル間の結合度行列
        '''
        # 準備
        self.size = size
        self.L = L
        self.mus = mus

        if sigma2s is None:
            self.sigma2s = np.full(L, 1.)
        elif isinstance(sigma2s, numbers.Number):
            self.sigma2s = np.full(L, sigma2s)
        else:
            self.sigma2s = sigma2s

        if isinstance(beta, numbers.Number):
            self.beta = beta*np.identity(L)
        else:
            self.beta = beta

    def MAPest(self, ys):
        '''
        MAP推定関数
        @param ys 観測値
        '''
        M, N = self.size
        Evens = np.asarray([[(m%2==0 and n%2==0) or (m%2==1 and n%2==1) for n in range(N)] for m in range(M)], dtype=bool).ravel() # 偶頂点のboolean index(M*N, )
        Odds = np.logical_not(Evens) # 奇頂点のboolean index(M*N, )

        logL = -(ys.ravel()[:, np.newaxis]-self.mus)**2/(2*self.sigma2s) # 対数尤度(M*N, L)
        logL_e = logL[Evens] # 対数尤度の偶頂点部分
        logL_o = logL[Odds] # 対数尤度の奇頂点部分

        #A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(nx.grid_2d_graph(M, N), ordering='sorted')) # 隣接行列(M*N, M*N)
        #A_eo = A[Evens][:, Odds] # 隣接行列の奇頂点->偶頂点部分
        #A_oe = A[Odds][:, Evens] # 隣接行列の偶頂点->奇頂点部分
        # 大きいnx.Graphを作ると止まるので，直接偶頂点x奇頂点の隣接行列を作る
        A_eo = compressed_adjacency_matrix(M, N)
        A_oe = A_eo.T

        X = logL.argmax(axis=1) # 現在のラベル(M*N, )
        Xold_e = X_e = X[Evens] # 現在のラベルの偶頂点部分
        Xold_o = X_o = X[Odds] # 現在のラベルの奇頂点部分

        maxiter = 100 # 最大反復回数

        for step in range(maxiter):
            print('step: {0}'.format(step))
            Xold_e, Xold_o = X_e, X_o

            # 偶頂点の更新
            nbr_labels = np.asarray([A_eo.dot((X_o==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            post = logL_e + nbr_labels.dot(self.beta) # 事後確率(M*N/2, L)
            X_e = post.argmax(axis=1)

            # 奇頂点の更新
            nbr_labels = np.asarray([A_oe.dot((X_e==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            post = logL_o + nbr_labels.dot(self.beta) # 事後確率(M*N/2, L)
            X_o = post.argmax(axis=1)

            # 停止判定
            if (X_e != Xold_e).sum() == 0 and (X_o != Xold_o).sum() == 0:
                # 更新されなくなった時終了
                break

        X[Evens] = X_e
        X[Odds] = X_o
        return X.reshape((M, N))

class cHMRF(object):
    '''
    continuous Hidden Markov Random Fieldクラス．
    マルコフ確率場はcontinuous intensities，観測はGaussian emission
    '''
    def __init__(self, size, tau2=10., sigma2=10.):
        '''
        コンストラクタ
        @param size MRFの大きさ(M, N)
        @param tau2 MRFの条件つき分散
        @param sigma2 観測分散
        '''
        # 準備
        self.size = size
        self.tau2 = tau2
        self.sigma2 = sigma2

    def MAPest(self, Y, tol=1e-3):
        '''
        MAP推定関数
        @param Y 観測値
        @param tol 許容誤差
        '''
        M, N = self.size
        Evens = np.asarray([[(m%2==0 and n%2==0) or (m%2==1 and n%2==1) for n in range(N)] for m in range(M)], dtype=bool).ravel() # 偶頂点のboolean index(M*N, )
        Odds = np.logical_not(Evens) # 奇頂点のboolean index(M*N, )

        #A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(nx.grid_2d_graph(M, N), ordering='sorted')) # 隣接行列(M*N, M*N)
        #A_eo = A[Evens][:, Odds] # 隣接行列の奇頂点->偶頂点部分
        #A_oe = A[Odds][:, Evens] # 隣接行列の偶頂点->奇頂点部分
        # 大きいnx.Graphを作ると止まるので，直接偶頂点x奇頂点の隣接行列を作る
        A_eo = compressed_adjacency_matrix(M, N)
        W_eo = normalize(A_eo, norm='l1', axis=1)
        W_oe = (normalize(A_eo, norm='l1', axis=0)).T

        Y_e = Y.ravel()[Evens]
        Y_o = Y.ravel()[Odds]

        X = np.zeros_like(Y) # 現在の潜在変数(M*N, )
        Xold_e = X_e = Y_e # 現在の潜在変数の偶頂点部分
        Xold_o = X_o = Y_o # 現在の潜在変数の奇頂点部分

        maxiter = 100 # 最大反復回数

        print('lets iter')
        for step in range(maxiter):
            print('step: {0}'.format(step))
            Xold_e, Xold_o = X_e, X_o

            # 偶頂点の更新
            X_e = (self.tau2*Y_e + self.sigma2*W_eo.dot(X_o))/(self.tau2 + self.sigma2)

            # 奇頂点の更新
            X_o = (self.tau2*Y_o + self.sigma2*W_oe.dot(X_e))/(self.tau2 + self.sigma2)

            # 停止判定
            if np.linalg.norm(X_e - Xold_e) + np.linalg.norm(X_o - Xold_o) < tol:
                # 更新されなくなった時終了
                break

        X.ravel()[Evens] = X_e
        X.ravel()[Odds] = X_o
        return X

class EHMRF(object):
    '''
    Extended Hidden Markov Random Fieldクラス．
    拡張マルコフ確率場はdiscrete colors，観測はGaussian emission
    '''
    def __init__(self, size, L, mus, sigma2s=None, alpha=None, beta=1.75):
        '''
        コンストラクタ
        @param size MRFの大きさ(M, N)
        @param L ラベル数
        @param mus: np.array((L, )) ラベルごとの観測平均
        @param sigma2s: np.array((L, )) ラベルごとの観測分散
        @param alpha: np.array((M, N, L)) 各頂点に各ラベルがつけられる度合い
        @param beta: np.array((L, L)) ラベル間の結合度行列
        '''
        # 準備
        self.size = size
        self.L = L
        self.mus = mus

        if sigma2s is None:
            self.sigma2s = np.full(L, 1.)
        elif isinstance(sigma2s, numbers.Number):
            self.sigma2s = np.full(L, sigma2s)
        else:
            self.sigma2s = sigma2s

        if alpha is None:
            self.alpha = np.zeros((M, N, L))
        else:
            self.alpha = alpha

        if isinstance(beta, numbers.Number):
            self.beta = beta*np.identity(L)
        else:
            self.beta = beta

    def MAPest(self, ys):
        '''
        MAP推定関数
        @param ys 観測値
        '''
        M, N = self.size
        Evens = np.asarray([[(m%2==0 and n%2==0) or (m%2==1 and n%2==1) for n in range(N)] for m in range(M)], dtype=bool).ravel() # 偶頂点のboolean index(M*N, )
        Odds = np.logical_not(Evens) # 奇頂点のboolean index(M*N, )

        logL_alpha = -(ys.ravel()[:, np.newaxis]-self.mus)**2/(2*self.sigma2s)+self.alpha.reshape((M*N, self.L)) # 対数尤度＋各頂点のprior(M*N, L)
        logL_alpha_e = logL_alpha[Evens] # 対数尤度＋各頂点のpriorの偶頂点部分
        logL_alpha_o = logL_alpha[Odds] # 対数尤度＋各頂点のpriorの奇頂点部分

        #A = nx.adjacency_matrix(nx.convert_node_labels_to_integers(nx.grid_2d_graph(M, N), ordering='sorted')) # 隣接行列(M*N, M*N)
        #A_eo = A[Evens][:, Odds] # 隣接行列の奇頂点->偶頂点部分
        #A_oe = A[Odds][:, Evens] # 隣接行列の偶頂点->奇頂点部分
        # 大きいnx.Graphを作ると止まるので，直接偶頂点x奇頂点の隣接行列を作る
        A_eo = compressed_adjacency_matrix(M, N)
        A_oe = A_eo.T

        X = logL_alpha.argmax(axis=1) # 現在のラベル(M*N, )
        Xold_e = X_e = X[Evens] # 現在のラベルの偶頂点部分
        Xold_o = X_o = X[Odds] # 現在のラベルの奇頂点部分

        maxiter = 100 # 最大反復回数

        for step in range(maxiter):
            print('step: {0}'.format(step))
            Xold_e, Xold_o = X_e, X_o

            # 偶頂点の更新
            nbr_labels = np.asarray([A_eo.dot((X_o==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            post = logL_alpha_e + nbr_labels.dot(self.beta) # 事後確率(M*N/2, L)
            X_e = post.argmax(axis=1)

            # 奇頂点の更新
            nbr_labels = np.asarray([A_oe.dot((X_e==l).astype(int)) for l in range(self.L)]).T # 頂点ごとの隣接ラベル数(M*N/2, L)
            post = logL_alpha_o + nbr_labels.dot(self.beta) # 事後確率(M*N/2, L)
            X_o = post.argmax(axis=1)

            # 停止判定
            if (X_e != Xold_e).sum() == 0 and (X_o != Xold_o).sum() == 0:
                # 更新されなくなった時終了
                break

        X[Evens] = X_e
        X[Odds] = X_o
        return X.reshape((M, N))

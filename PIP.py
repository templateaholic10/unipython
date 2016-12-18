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

def compressed_adjacency_matrix(size):
    '''
    圧縮した隣接行列を返す関数
    @param size: (Rows, Cols)
    @return scipy.sparse.lil_matrix((偶頂点, 奇頂点))
    '''
    Rows, Cols = size
    Ve, Vo = (Rows*Cols+1)//2, Rows*Cols//2 # 偶頂点と奇頂点の数
    A_eo = lil_matrix((Ve, Vo), dtype=int)

    ote = lambda ve: ve-(Cols+1)//2
    eto = lambda vo: vo-Cols//2
    ole = lambda ve: ve-int((Cols%2!=0)or(ve//(Cols//2))%2==0)
    elo = lambda vo: vo-int((Cols%2==0)and(vo//(Cols//2))%2==1)

    # 偶頂点を舐めて左または上の奇頂点に線を引く
    for ve in range(1, Ve):
        if ve < (Cols+1)//2:
            A_eo[ve, ole(ve)] = 1 # 上辺上ならば左のみ
        elif ve%Cols == 0:
            A_eo[ve, ote(ve)] = 1 # 左辺上ならば上のみ
        else:
            A_eo[ve, ole(ve)] = A_eo[ve, ote(ve)] = 1

    # 奇頂点を舐めて左または上の偶頂点に線を引く
    for vo in range(Vo):
        if vo < Cols//2:
            A_eo[elo(vo), vo] = 1 # 上辺上ならば左のみ
        elif vo%Cols == Cols//2:
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
    def __init__(self, size, L, beta=1.75):
        '''
        コンストラクタ
        @param size: (Rows, Cols) MRFの大きさ
        @param L ラベル数
        @param beta ラベル間の結合度行列
        '''
        # 準備
        self.size = size
        self.Rows, self.Cols = self.size
        self.L = L

        if isinstance(beta, numbers.Number):
            self.beta = beta*np.identity(L)
        else:
            self.beta = beta

        self.Evens = np.asarray([[(row%2==0 and col%2==0) or (row%2==1 and col%2==1) for col in range(self.Cols)] for row in range(self.Rows)], dtype=bool).ravel() # 偶頂点のboolean index(Rows*Cols, )
        self.Odds = np.logical_not(self.Evens) # 奇頂点のboolean index(Rows*Cols, )
        self.A_eo = compressed_adjacency_matrix(self.size) # 偶頂点と奇頂点の隣接行列

    def MAPest(self, ys, mus, sigma2s):
        '''
        MAP推定関数
        @param ys 観測値
        @param mus ラベルごとの観測平均
        @param sigma2s ラベルごとの観測分散
        @return MAP推定値と事後確率のタプル
        '''
        logL = -(ys.ravel()[:, np.newaxis]-mus)**2/(2*sigma2s) # 対数尤度(Rows*Cols, L)
        logL_e = logL[self.Evens] # 対数尤度の偶頂点部分
        logL_o = logL[self.Odds] # 対数尤度の奇頂点部分

        X = logL.argmax(axis=1) # 現在のラベル(Rows*Cols, )
        Xold_e = X_e = X[self.Evens] # 現在のラベルの偶頂点部分
        Xold_o = X_o = X[self.Odds] # 現在のラベルの奇頂点部分

        maxiter = 100 # 最大反復回数

        for step in range(maxiter):
            print('step: {0}'.format(step))
            Xold_e, Xold_o = X_e, X_o

            # 偶頂点の更新
            nbr_labels = self.A_eo.dot((X_o[:, np.newaxis]==np.arange(self.L)).astype(int)) # 頂点ごとの隣接ラベル数(Rows*Cols/2, L)
            logjoint_e = logL_e + nbr_labels.dot(self.beta) # 対数同時確率＝対数事後確率＋定数(Rows*Cols/2, L)
            X_e = logjoint_e.argmax(axis=1)

            # 奇頂点の更新
            nbr_labels = self.A_eo.T.dot((X_e[:, np.newaxis]==np.arange(self.L)).astype(int)) # 頂点ごとの隣接ラベル数(Rows*Cols/2, L)
            logjoint_o = logL_o + nbr_labels.dot(self.beta) # 対数同時確率＝対数事後確率＋定数(Rows*Cols/2, L)
            X_o = logjoint_o.argmax(axis=1)

            # 停止判定
            if (X_e != Xold_e).sum() == 0 and (X_o != Xold_o).sum() == 0:
                # 更新されなくなった時終了
                break

        X[self.Evens] = X_e
        X[self.Odds] = X_o
        joint = np.zeros_like(logL) # 同時確率(Rows*Cols, L)
        joint[self.Evens] = np.exp(logjoint_e)
        joint[self.Odds] = np.exp(logjoint_o)
        return X.reshape(self.size), (joint/joint.sum(axis=1, keepdims=True)).reshape(self.size + (self.L,))

    def joint(self, ys, mus, sigma2s):
        '''
        同時確率を返す関数
        @param ys 観測値
        @param mus ラベルごとの観測平均
        @param sigma2s ラベルごとの観測分散
        @return 同時確率
        '''
        logL = -(ys.ravel()[:, np.newaxis]-mus)**2/(2*sigma2s) # 対数尤度(Rows*Cols, L)
        logL_e = logL[self.Evens] # 対数尤度の偶頂点部分
        logL_o = logL[self.Odds] # 対数尤度の奇頂点部分

        X = logL.argmax(axis=1) # 現在のラベル(Rows*Cols, )
        Xold_e = X_e = X[self.Evens] # 現在のラベルの偶頂点部分
        Xold_o = X_o = X[self.Odds] # 現在のラベルの奇頂点部分

        maxiter = 100 # 最大反復回数

        for step in range(maxiter):
            print('step: {0}'.format(step))
            Xold_e, Xold_o = X_e, X_o

            # 偶頂点の更新
            nbr_labels = self.A_eo.dot((X_o[:, np.newaxis]==np.arange(self.L)).astype(int)) # 頂点ごとの隣接ラベル数(Rows*Cols/2, L)
            logjoint_e = logL_e + nbr_labels.dot(self.beta) # 対数同時確率＝対数事後確率＋定数(Rows*Cols/2, L)
            X_e = logjoint_e.argmax(axis=1)

            # 奇頂点の更新
            nbr_labels = self.A_eo.T.dot((X_e[:, np.newaxis]==np.arange(self.L)).astype(int)) # 頂点ごとの隣接ラベル数(Rows*Cols/2, L)
            logjoint_o = logL_o + nbr_labels.dot(self.beta) # 対数同時確率＝対数事後確率＋定数(Rows*Cols/2, L)
            X_o = logjoint_o.argmax(axis=1)

            # 停止判定
            if (X_e != Xold_e).sum() == 0 and (X_o != Xold_o).sum() == 0:
                # 更新されなくなった時終了
                break

        retval = np.zeros_like(logL) # 同時確率(Rows*Cols, L)
        retval[self.Evens] = np.exp(logjoint_e)
        retval[self.Odds] = np.exp(logjoint_o)
        return retval

    def EM(self, ys, mus0, sigma2s0):
        '''
        EMアルゴリズムで(mu_l, sigma2_l)を推定する関数
        @param ys 観測値
        @param mus0 ラベルごとの観測平均の初期値
        @param sigma2s0 ラベルごとの観測分散の初期値
        @return パラメータと事後確率のタプル
        '''
        mus, sigma2s = mus0, sigma2s0

        maxiter = 100 # 最大反復回数
        tol = 1e-3 # 許容誤差

        print('STEP: 0')
        # E-step
        joint = self.joint(ys, mus, sigma2s) # 同時確率

        # 停止判定
        F = -np.inf
        print('F: ', F)

        # M-step
        post = joint/joint.sum(axis=1, keepdims=True)
        postsum = post.sum(axis=0)
        mus = ys.ravel().dot(post)/postsum
        sigma2s = ((ys.ravel()[:, np.newaxis]-mus)**2*post).sum(axis=0)/postsum

        for step in range(1, maxiter):
            print('STEP: {0}'.format(step))
            Fold = F
            # E-step
            joint = self.joint(ys, mus, sigma2s) # 同時確率

            # 停止判定
            logjoint = np.zeros_like(joint)
            logjoint[post!=0] = np.log(joint[post!=0])
            F = (post*logjoint).sum()
            print('F: ', F)
            if F - Fold < 0:
                # 対数尤度が増大している
                print('likelihood bigger')
                break
            elif F - Fold < tol:
                # 対数尤度が上がらない
                print('likelihood maximal')
                break

            # M-step
            post = joint/joint.sum(axis=1, keepdims=True)
            postsum = post.sum(axis=0)
            mus = ys.ravel().dot(post)/postsum
            sigma2s = ((ys.ravel()[:, np.newaxis]-mus)**2*post).sum(axis=0)/postsum

        post = joint/joint.sum(axis=1, keepdims=True)
        return (mus, sigma2s), post.reshape(self.size)

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

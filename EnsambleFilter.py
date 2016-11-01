# -*- coding: utf-8 -*-

"""
アンサンブルフィルタ系列のクラスを提供するモジュール
"""

import numpy as np
import exnumpy as exnp

class EnKF(object):
    """
    アンサンブルカルマンフィルタ
    x_{t+1} = f(x_t, v) : システムモデル
    y_t = Hx_t + w, w~N(0, R) : 観測モデル
    """

    def __init__(self, f, H, R):
        """
        コンストラクタ
        @param f ユニバーサル遷移関数
        @param H 観測行列
        @param R 観測ノイズの共分散行列
        """
        self.f = f
        self.H = H
        self.R = R
        self.q, self.p = H.shape

    def forecast(self, xs):
        """
        予測
        @param xs アンサンブル
        @return 予測アンサンブル
        """
        return self.f(xs)

    def analyze(self, xs, y):
        """
        解析
        @param xs アンサンブル
        @param y 観測値
        @return 解析アンサンブル
        """
        M, _ = xs.shape
        # アンサンブル平均
        xm = xs.mean(axis=0)
        # アンサンブル不偏分散
        P = (xs - xm).T.dot(xs - xm)/(M-1)
        # カルマンゲイン
        K = P.dot(self.H.T).dot(np.linalg.inv(self.H.dot(P).dot(self.H.T) + self.R))

        return xs + exnp.mxvs(K, y - exnp.mxvs(self.H, xs) + np.random.multivariate_normal(np.zeros(self.q), self.R, M))

    def run(self, x0s, yseq, Nout):
        """
        計算
        @param x0s 初期アンサンブル
        @param yseq 観測値列
        @param Nout 観測周期
        @return アンサンブル列
        """
        M, _ = x0s.shape
        Nobs, _ = yseq.shape
        xsseq = np.zeros((Nobs*Nout+1, M, self.p))

        # 初期化
        xs = x0s
        xsseq[0] = xs

        for nobs in range(Nobs):
            # 予測
            for n in range(nobs*Nout, (nobs+1)*Nout):
                xs = self.forecast(xs)
                xsseq[n+1] = xs

            # 解析
            xs = self.analyze(xs, yseq[nobs])
            xsseq[(nobs+1)*Nout] = xs

        return xsseq

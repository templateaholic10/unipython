# -*- coding: utf-8 -*-

'''
サンプルシステムモデル
'''

import numpy as np

class Oscillator(object):
    '''
    振動子モデル．ノイズなしの２次元システムモデル
    \ddot\theta+2\gamma\dot\theta+\omega^2\theta=0
    '''

    dim = 2

    def __init__(self, omega=np.pi, gamma=0, dt=0.1):
        '''
        コンストラクタ
        @param omega 角振動数
        @param gamma 減衰係数
        @param dt 離散化幅
        '''
        self.omega = omega
        self.gamma = gamma
        self.dt = dt

    def ref(self, x0=np.array([1, 0]), N=1000):
        '''
        リファレンストラジェクトリ
        @param x0 初期状態 np.array([\theta_0, \dot\theta_0])
        @param N ステップ数
        @return 状態列
        '''
        t = np.array(range(N))*self.dt
        if self.gamma > self.omega:
            # 過減衰
            alpha = np.sqrt(self.gamma**2-self.omega**2)

            C1 = x0[0]
            C2 = (self.gamma*x0[0]+x0[1])/alpha
            theta = np.exp(-self.gamma*t)*(C1*np.cosh(alpha*t)+C2*np.sinh(alpha*t))

            C1 = x0[1]
            C2 = -(self.gamma*x0[1]+self.omega**2*x0[0])/alpha
            dtheta = np.exp(-self.gamma*t)*(C1*np.cosh(alpha*t)+C2*np.sinh(alpha*t))

        elif self.gamma < self.omega:
            # 減衰振動
            alpha = np.sqrt(self.omega**2-self.gamma**2)

            C1 = x0[0]
            C2 = (self.gamma*x0[0]+x0[1])/alpha
            theta = np.exp(-self.gamma*t)*(C1*np.cos(alpha*t)+C2*np.sin(alpha*t))

            C1 = x0[1]
            C2 = -(self.gamma*x0[1]+self.omega**2*x0[0])/alpha
            dtheta = np.exp(-self.gamma*t)*(C1*np.cos(alpha*t)+C2*np.sin(alpha*t))

        else:
            # 臨界減衰
            theta = np.exp(-self.omega*t)*(x0[0]+(x0[1]+self.omega*x0[0])*t)
            dtheta = np.exp(-self.omega*t)*(x0[1]-self.omega*(x0[1]+self.omega*x0[0])*t)

        return np.stack((theta, dtheta), axis=-1)

    def __f(self, x):
        '''
        微分方程式の標準形の右辺
        @param x 状態
        @return dx = f(x)
        '''
        return np.array([x[1], -2*self.gamma*x[1]-self.omega**2*x[0]])

    def forward(self, x, method='Runge-Kutta'):
        '''
        フォワード計算
        @param x 現在の状態
        @param method 差分化方法 ['Runge-Kutta', 'explicit-Euler']
        @return １ステップ先の状態
        '''
        if method == 'Runge-Kutta':
            k1 = self.__f(x)
            k2 = self.__f(x+self.dt/2*k1)
            k3 = self.__f(x+self.dt/2*k2)
            k4 = self.__f(x+self.dt*k3)
            return x+self.dt/6*(k1+2*k2+2*k3+k4)

        elif method == 'explicit-Euler':
            return x+self.dt*self.__f(x)

    def simulate(self, x0=np.array([1, 0]), N=1000, method='Runge-Kutta'):
        '''
        シミュレーション
        @param x0 初期状態 np.array([\theta_0, \dot\theta_0])
        @param N ステップ数
        @param method 差分化方法 ['Runge-Kutta', 'explicit-Euler']
        @return 状態列
        '''
        xseq = np.zeros((N, Oscillator.dim))
        xseq[0] = x = x0
        for n in range(1, N):
            xseq[n] = x = self.forward(x, method)
        return xseq

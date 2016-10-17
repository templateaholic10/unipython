# -*- cofing: utf-8 -*-

"""
# exwaveモジュール

wave及びpyaudioのラッパーモジュール．
wavファイル<>バイナリ<>numpy.arrayを行き来することができる．

waveのラッパーとしての機能
- wavファイルをjsonやpickleと同様にやりとりできるdump，load関数
pyaudioのラッパーとしての機能
- play関数
"""

import numpy as np
import wave
import pyaudio
import copy
import struct

numpy_types = {8: 'int8', 16: 'int16', 32: 'int32'}
byte_formats = {8: 'b', 16: 'h', 32: 'i'}
audio_formats = {8: pyaudio.paInt8, 16: pyaudio.paInt16, 32: pyaudio.paInt32}
default_bit = 16

def clip(data, maxim=1):
    """
    クリッピング関数
    @param data 信号
    @param maxim クリッピング振幅
    @return 信号
    """
    retval = copy.deepcopy(data)
    retval[data > maxim] = maxim
    retval[data < -maxim] = -maxim
    return retval

def dumpb(data, bit=default_bit, maxim=None):
    """
    バイナリデータを作る関数
    @param data numpy.array
    @param bit ビット数
    @param maxim これが最大値になるよう規格化する．デフォルトは信号の最大値
    @return バイナリデータ
    """
    if bit not in byte_formats:
        return None
    if maxim is None:
        maxim = data.max()
    # クリップ
    cliped = clip(data, maxim)
    # 整数化
    maxint = 2**(bit-1)-1
    inted = (cliped/maxim*maxint).astype(int)
    # バイナリ化
    return struct.pack(byte_formats[bit]*len(inted), *list(inted))

def dump(data, fs, fp, bit=default_bit, maxim=None):
    """
    wavファイルに出力する関数
    @param data numpy.array
    @param fs サンプリング周波数
    @param fp ファイルポインタ
    @param bit ビット数
    @param maxim これが最大値になるよう規格化する．デフォルトは信号の最大値
    """
    binary = dumpb(data, bit, maxim)
    fp.setnchannels(1)
    fp.setsampwidth(int(bit/8))
    fp.setframerate(fs)
    fp.writeframes(binary)

def loadb(binary, bit=default_bit):
    """
    バイナリデータをnumpy.arrayにする関数
    @param binary バイナリデータ
    @param bit ビット数
    @return [-1, 1]のnumpy.array
    """
    maxint = 2**(bit-1)-1
    return np.frombuffer(binary, numpy_types[bit])/maxint

def load(fp, bit=default_bit):
    """
    wavファイルからnumpy.arrayを作る関数
    @param fp ファイルポインタ
    @param bit ビット数
    @return [-1, 1]のnumpy.arrayとサンプル周波数のタプル
    """
    maxint = 2**(bit-1)-1
    fs = fp.getframerate()
    binary = fp.readframes(fp.getnframes())
    return loadb(binary), fs

def playb(binary, fs, bit=default_bit):
    """
    バイナリデータを再生する関数
    @param binary バイナリデータ
    @param fs サンプリング周波数
    @param bit ビット数
    """
    # ストリームを開く
    p = pyaudio.PyAudio()
    stream = p.open(format=audio_formats[bit], channels=1, rate=int(fs), output=True)
    # チャンク単位でストリームに出力
    chunk = 1024
    sp = 0 # 再生位置ポインタ
    buffer = binary[sp:sp+chunk]
    while buffer != b'':
        stream.write(buffer)
        sp += chunk
        buffer = binary[sp:sp+chunk]
    stream.close()
    p.terminate()

def play(data, fs, bit=default_bit, maxim=None):
    """
    numpy.arrayを再生する関数
    @param data np.array
    @param fs サンプリング周波数
    @param bit ビット数
    @param maxim これが最大値になるよう規格化する．デフォルトは信号の最大値
    """
    binary = dumpb(data, bit, maxim)
    playb(binary, fs, bit)

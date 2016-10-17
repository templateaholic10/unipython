# -*- coding: utf-8 -*-

"""
# dtmモジュール

簡単な曲データを生成できるモジュール．
"""

import numpy as np
import math

# 音名: 周波数[Hz]辞書を作成
keys = ['C', 'Cs', 'D', 'Ds', 'E', 'F', 'Fs', 'G', 'Gs', 'A', 'As', 'B']
keynum = len(keys)

octnum = 10

pitches = {'none': 0}

baseoct = 4
basekeyid = keys.index('A')
basefreq = 440
for octave in range(octnum):
    for keyid, key in enumerate(keys):
        pitches[key+str(octave)] = basefreq*2**(octave-baseoct+(keyid-basekeyid)/keynum)

# 簡易音名を追加
pitches['Do'] = pitches['C4']
pitches['Re'] = pitches['D4']
pitches['Mi'] = pitches['E4']
pitches['Fa'] = pitches['F4']
pitches['Sol'] = pitches['G4']
pitches['La'] = pitches['A4']
pitches['Si'] = pitches['B4']
pitches['do'] = pitches['C5']
pitches['re'] = pitches['D5']
pitches['mi'] = pitches['E5']
pitches['fa'] = pitches['F5']
pitches['sol'] = pitches['G5']
pitches['la'] = pitches['A5']
pitches['si'] = pitches['B5']

def pure_tone(f, fs=8000, duration=1):
    """
    純音を作る関数
    @param f 周波数[Hz]
    @param fs サンプリング周波数[Hz]
    @param duration 持続時間[s]
    @return numpy.array
    """
    ts = np.arange(0, duration, 1/fs)
    return np.sin(2*np.pi*f*ts)

class Translater:
    """
    楽譜を波形データに翻訳するクラス
    """
    def __init__(self, fs, bpm):
        """
        コンストラクタ
        @param fs サンプリング周波数
        @param bpm Beats per Minute
        """
        self.fs = fs
        self.bpm = bpm
        self.samplePerBeat = self.fs*60/self.bpm

    def note(self, keyname, beat=1):
        """
        単音符を翻訳する関数
        @param keyname 音名
        @param beat 拍数
        @return numpy.array
        """
        return pure_tone(pitches[keyname], self.fs, beat/self.bpm*60)

    def score(self, notelist):
        """
        楽譜を翻訳する関数
        @param notelist 音符リスト．各音符は(音名,拍数,開始拍)のタプル
        @return numpy.array
        """
        maxbeat = max(enumerate(notelist), key=lambda x: x[1][1]+x[1][2])[1]
        wave = np.zeros(math.ceil(maxbeat*self.samplePerBeat))
        for tpl in notelist:
            keyname, beat, firstbeat = tpl
            note = self.note(keyname, beat)
            offset = math.ceil(firstbeat*self.samplePerBeat)
            wave[offset:offset+len(note)] = note

        return wave

    def melody(self, notelist):
        """
        メロディを翻訳する関数．scoreと異なり，同時に1音しか鳴らないので開始拍を指定しなくて良い
        @param notelist 音符リスト．各音符は(音名,拍数)のタプル
        @return numpy.array
        """
        maxbeat = sum([tpl[1] for tpl in notelist])
        wave = np.zeros(math.ceil(maxbeat*self.samplePerBeat))
        head = 0
        for tpl in notelist:
            keyname, beat = tpl
            note = self.note(keyname, beat)
            offset = math.ceil(head*self.samplePerBeat)
            wave[offset:offset+len(note)] = note
            head += beat

        return wave

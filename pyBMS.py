# -*- coding: utf-8 -*-
'''
BMSフォーマットのインタフェース

CNNでBMS譜面の難易度を推定するために作成しました．
例えば以下のプログラムはBMSを時間解像度0.05秒で画像化します：

bms = BMS(filename)
arr = note2array(*bms.score.tosec(), 0.05)

この画像を入力，難易度を出力としてCNNで学習すると興味深かったです．
'''

import numpy as np
from collections import Counter

_num_lanes = 8

def _beats2nodim(beats, chlength=1.):
    '''
    １小節の拍リストを無次元時刻リストに変換する内部関数
    @param beats 拍リスト
    @param chlength 小節長の倍率
    @return 時刻とコードの組のリスト
    '''
    return [(ix/len(beats)*chlength, code) for ix, code in enumerate(beats) if code!=0]

class Bar(object):
    '''
    小節クラス
    @attr notes レーンごとの拍リストリスト
    @attr chlength 小節長の倍率
    @attr bpm BPM
    @attr chbpm 拍リストリスト形式のBPM変化
    '''
    def __init__(self):
        self.notes = [[] for laneid in range(_num_lanes)]
        self.chlength = 1.
        self.bpm = 130.
        self.chbpm = []

    def tonodim(self):
        '''
        小節を無次元時刻で表す関数
        @return レーンごとの無次元時刻リストリスト
        '''
        # 無次元量に変換
        notes_nodim = [[] for laneid in range(_num_lanes)]
        for laneid in range(_num_lanes):
            notes_nodim[laneid] = sorted([tpl for beats in self.notes[laneid] for tpl in _beats2nodim(beats, self.chlength)], key=lambda tpl: tpl[0])
        return notes_nodim, self.chlength

    def tosec(self):
        '''
        小節を秒で表す関数
        @return レーンごとの秒リストリスト
        '''
        # 無次元量に変換
        notes_nodim = [[] for laneid in range(_num_lanes)]
        for laneid in range(_num_lanes):
            notes_nodim[laneid] = sorted([tpl for beats in self.notes[laneid] for tpl in _beats2nodim(beats, self.chlength)], key=lambda tpl: tpl[0])
        chbpm_nodim = sorted([elem for beats in self.chbpm for elem in _beats2nodim(beats, self.chlength)], key=lambda tpl: tpl[0])
        tmp1, tmp2 = [[tpl[ix] for tpl in chbpm_nodim] for ix in [0, 1]]
        endbpm_nodim = zip(tmp1+[1.], [self.bpm]+tmp2)

        # 実時刻に変換
        notes_sec = [[] for laneid in range(_num_lanes)]
        start_sec = 0.
        start_nodim = 0.
        for end_nodim, bpm in endbpm_nodim:
            sec_per_bar = 4.*60./bpm
            for laneid in range(_num_lanes):
                notes_sec[laneid] += [(start_sec+(nodim-start_nodim)*sec_per_bar, code) for nodim, code in notes_nodim[laneid] if start_nodim<=nodim<end_nodim]
            start_sec = start_sec+(end_nodim-start_nodim)*sec_per_bar
            start_nodim = end_nodim

        return notes_sec, start_sec

class Score(object):
    '''
    小節クラス．Barオブジェクトリストのラッパー．
    Scoreのスライスはより小さなScoreオブジェクトを返す
    @attr data 実体のBarクラスリスト
    '''
    def __init__(self, data):
        self.data = data

    def __getitem__(self, ix):
        if isinstance(ix, slice):
            return Score(self.data[ix])
        else:
            return self.data[ix]

    def __len__(self):
        return self.data.__len__()

    def tonodim(self):
        '''
        譜面を無次元時刻で表す関数
        @return レーンごとの無次元時刻リストリスト
        '''
        notes_nodim = [[] for laneid in range(_num_lanes)]
        start_nodim = 0.
        for barid in range(len(self.data)):
            _notes_nodim, length_nodim = self.data[barid].tonodim()
            for laneid in range(_num_lanes):
                notes_nodim[laneid] += [(start_nodim+nodim, code) for nodim, code in _notes_nodim[laneid]]
            start_nodim += length_nodim
        return notes_nodim, start_nodim

    def tosec(self):
        '''
        譜面を秒で表す関数
        @return レーンごとの秒リストリスト
        '''
        notes_sec = [[] for laneid in range(_num_lanes)]
        start_sec = 0.
        for barid in range(len(self.data)):
            _notes_sec, length_sec = self.data[barid].tosec()
            for laneid in range(_num_lanes):
                notes_sec[laneid] += [(start_sec+sec, code) for sec, code in _notes_sec[laneid]]
            start_sec += length_sec
        return notes_sec, start_sec

class BMS(object):
    '''
    BMSクラス
    @attr metadata メタデータ辞書
    @attr wavs WAVファイル辞書
    @attr midi MIDIファイル
    @attr bmps BMPファイル辞書
    @attr score Scoreオブジェクト
    '''
    def _process_command(self, line):
        '''
        BMSファイルの行を処理する内部関数
        @param line 行
        '''
        # コマンドの判定
        if line[0]!='#':
            return

        stripped = line[1:].rstrip()

        # ヘッダコマンドの判定
        tmp = stripped.split(' ')
        command, param_str = tmp[0], ' '.join(tmp[1:])
        wavid_base = 36
        bmpid_base = 36
        if command in ['GENRE', 'TITLE', 'ARTIST']:
            self.metadata[command] = param_str
        elif command in ['PLAYER', 'PLAYLEVEL', 'RANK', 'VOLWAV']:
            self.metadata[command] = int(param_str)
        elif command in ['BPM', 'TOTAL']:
            self.metadata[command] = float(param_str)
        elif command[:3]=='WAV':
            self.wavs[int(command[3:], wavid_base)] = param_str
        elif command=='MIDIFILE':
            self.midi = param_str
        elif command[:3]=='BMP':
            self.bmps[int(command[3:], bmpid_base)] = param_str

        # メインコマンドの判定
        tmp = stripped.split(':')
        command, param_str = tmp[0], ':'.join(tmp[1:])
        if len(command)!=5 or not command.isdigit():
            return

        barid, channel = int(command[:3]), int(command[3:])
        meta_channels = {
            1: 'bgm',
            2: 'chlength',
            3: 'chbpm',
            4: 'bga',
            5: 'chobj',
            6: 'chmiss',
            7: 'splite'
        }
        note_channels = {
            11: 1,
            12: 2,
            13: 3,
            14: 4,
            15: 5,
            16: 0,
            18: 6,
            19: 7
        }
        bpm_base = 16
        if barid not in self._bar_dict:
            self._bar_dict[barid] = Bar()
        if channel in note_channels:
            laneid = note_channels[channel]
            num_beats = len(param_str)//2
            self._bar_dict[barid].notes[laneid].append([int(param_str[2*beatid:2*(beatid+1)], wavid_base) for beatid in range(num_beats)])
        elif channel in meta_channels:
            if meta_channels[channel]=='chlength':
                self._bar_dict[barid].chlength = float(param_str)
            elif meta_channels[channel]=='chbpm':
                num_beats = len(param_str)//2
                self._bar_dict[barid].chbpm.append([int(param_str[2*beatid:2*(beatid+1)], bpm_base) for beatid in range(num_beats)])

    def __init__(self, finname):
        # bms/bmeファイルの読み込み
        self.metadata = dict()
        self.wavs = dict()
        self.midi = None
        self.bmps = dict()
        self._bar_dict = dict()
        with open(finname, 'r') as fin:
            for line in fin:
                self._process_command(line)

        maxbar = max(self._bar_dict)

        # bpmの設定
        bpm = self.metadata['BPM']
        for barid in range(maxbar+1):
            if barid not in self._bar_dict:
                self._bar_dict[barid] = Bar()
            self._bar_dict[barid].bpm = bpm
            if self._bar_dict[barid].chbpm:
                _, bpm = max([elem for beats in self._bar_dict[barid].chbpm for elem in _beats2nodim(beats)], key=lambda tpl: tpl[0])

        # Scoreの作成
        self.score = Score([self._bar_dict[barid] for barid in sorted(self._bar_dict)])

def notes2array(notes, end_time, time_resolusion=1./16.):
    '''
    時刻で表現した譜面を画像に変換する関数
    @param notes レーンごとの時刻リストリスト
    @param end_time 終了時刻
    @param time_resolusion 時間解像度
    @return レーン数xフレーム数のnumpy.ndarray
    '''
    num_units = np.ceil(end_time/time_resolusion).astype(int)
    arr = np.zeros((_num_lanes, num_units), dtype=int)
    for laneid in range(_num_lanes):
        c = Counter([int(time/time_resolusion) for time, code in notes[laneid]])
        for ix in c:
            arr[laneid, ix] = c[ix]
    return arr

def notes2image(notes, end_time, shape, note_weight=1):
    '''
    時刻で表現した譜面を観賞用の画像に変換する関数
    @param notes レーンごとの時刻リストリスト
    @param end_time 終了時刻
    @param shape 画像サイズ
    @param note_weight ノーツの厚さ
    @return shapeサイズのnumpy.ndarray
    '''
    Rows, Cols = shape
    img = np.zeros((Rows, Cols), dtype=np.uint8)
    for laneid in range(_num_lanes):
        for time, code in notes[laneid]:
            rowmin, rowmax = int(time/end_time*Rows), int(time/end_time*Rows)+note_weight+1
            colmin, colmax = int(laneid/_num_lanes*Cols), int((laneid+1)/_num_lanes*Cols)
            img[rowmin:rowmax, colmin:colmax] = 255
    return img

if __name__=='__main__':
    pass
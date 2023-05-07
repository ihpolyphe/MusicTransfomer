import numpy as np

import magenta.music as mm
from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel
from magenta.music.sequences_lib import concatenate_sequences

import note_seq

BATCH_SIZE = 4  # 一度に扱うデータ数
Z_SIZE = 512  # 潜在変数の数
TOTAL_STEPS = 512  # コードのベクトル化に使用
CHORD_DEPTH = 49  # コードのベクトル化に使用
SEQ_TIME = 2.0  # 各NoteSequenceの長さ

def trim(seqs, seq_time=SEQ_TIME):  # NoteSequenceの長さを揃える
    for i in range(len(seqs)):
        seqs[i] = mm.extract_subsequence(seqs[i], 0.0, seq_time)
        seqs[i].total_time = seq_time

def encode_chord(chord):  # コードの文字列をベクトルに変換
    index = mm.TriadChordOneHotEncoding().encode_event(chord)
    encoded = np.zeros([TOTAL_STEPS, CHORD_DEPTH])
    encoded[0,0] = 1.0
    encoded[1:,index] = 1.0
    return encoded

def set_instruments(note_sequences):  # 楽器の調整
    for i in range(len(note_sequences)):
        for note in note_sequences[i].notes:
            if note.is_drum:
                note.instrument = 9

# コードをラベルとしたConditional VAEのモデルを読み込みます。
config = configs.CONFIG_MAP["hier-multiperf_vel_1bar_med_chords"]
model = TrainedModel(
    config,
    batch_size=BATCH_SIZE,
    checkpoint_dir_or_path="/mnt/c/Users/hayat/Desktop/myself/models/music_vae/model_chords_fb64.ckpt")

#潜在変数から曲をデコードする際に、コードをラベルとして入力します。
#これにより、そのコードをベースにした曲のNoteSeqenceを生成することができます。

chord_1 = "Dm"
chord_2 = "F"
chord_3 = "Am"
chord_4 = "G"
chords = [chord_1, chord_2, chord_3, chord_4]
num_bars = 32
temperature = 0.25
z1 = np.random.normal(size=[Z_SIZE])
z2 = np.random.normal(size=[Z_SIZE])
z = np.array([z1+z2*t for t in np.linspace(0, 1, num_bars)])  # z1とz2の間を線形補間
seqs = [
    model.decode(
        length=TOTAL_STEPS,
        z=z[i:i+1, :],
        temperature=temperature,
        c_input=encode_chord(chords[i%4])
        )[0]
    for i in range(num_bars)
]

trim(seqs)
set_instruments(seqs)
seq = concatenate_sequences(seqs)

# エクスポート
note_seq.sequence_proto_to_midi_file(seq, "conditional_vae_excersize.mid")  #MIDI　データに変換し保存
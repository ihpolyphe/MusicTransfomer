import numpy as np
# from google.colab import files
import tensorflow.compat.v1 as tf
import datetime
from tensor2tensor import models
from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq
"""
入力midに対して伴奏を付与するスクリプト
"""

tf.disable_v2_behavior()

# モデルへ入力を渡す関数（generator）
inputs = []
decode_length = 0
def input_generator():
  global inputs
  while True:
    yield {
        "inputs": np.array([[inputs]], dtype=np.int32),
        "targets": np.zeros([1, 0], dtype=np.int32),
        "decode_length": np.array(decode_length, dtype=np.int32)
    }

# モデルの出力がidなので、それをMIDIにデコードする関数
def decode(ids, encoder):
    ids = list(ids)
    if text_encoder.EOS_ID in ids:
        ids = ids[:ids.index(text_encoder.EOS_ID)]
    return encoder.decode(ids)

model_name = "transformer"  # モデル
hparams_set = "transformer_tpu"  # ハイパーパラメータ
ckpt_path = "<checkpoint_path>"  # チェックポイント

# エンコーダー生成用のクラス
class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = MelodyToPianoPerformanceProblem()
encoders = problem.get_feature_encoders()

# ハイパーパラメータの設定
hparams = trainer_lib.create_hparams(hparams_set=hparams_set)
trainer_lib.add_problem_hparams(hparams, problem)
hparams.num_hidden_layers = 16  # 中間層の数
hparams.sampling_method = "random"  # サンプリング方法をランダムに

# デコーダーのハイパーパラメータを設定
decode_hparams = decoding.decode_hparams()
decode_hparams.alpha = 0.0
decode_hparams.beam_size = 1

# モデル（推定器）を構築
run_config = trainer_lib.create_run_config(hparams)
estimator = trainer_lib.create_estimator(
    model_name,
    hparams,
    run_config,
    decode_hparams=decode_hparams
    )

# 推定
input_fn = decoding.make_input_fn_from_generator(input_generator())  # 入力を生成する関数
predicted = estimator.predict(
    input_fn,
    checkpoint_path=ckpt_path  # チェックポイントを読み込む
    )

# 最初の推定結果は飛ばす
next(predicted)

# MIDIファイルの読み込み
input_midi_path = "<midi_path>"

melody_seq = note_seq.midi_file_to_note_sequence(input_midi_path)

melody_instrument = note_seq.infer_melody_for_sequence(melody_seq)  # メロディを推定
notes = [note for note in melody_seq.notes if note.instrument==melody_instrument]  # メロディを抽出

melody_seq.notes.extend(
    sorted(notes, key=lambda note: note.start_time)  # noteを開始時刻順にソート
    )

inputs = encoders["inputs"].encode_note_sequence(melody_seq)  # NoteSequenceを入力用にエンコード
decode_length = 4096

# 推定結果をidとして取得
predicted_ids = next(predicted)["outputs"]

# idをNoteSequenceに変換
midi_file = decode(
    predicted_ids,
    encoder=encoders["targets"]
    )
seq = note_seq.midi_file_to_note_sequence(midi_file)

now = datetime.datetime.now()
filename = "result/accompaniment_" + now.strftime('%Y_%m_%d_%H_%M_%S') + '.mid'
note_seq.sequence_proto_to_midi_file(seq, filename)
import numpy as np
# from google.colab import files
import tensorflow as tf

#from tensor2tensor import models
#from tensor2tensor import problems
from tensor2tensor.data_generators import text_encoder
from tensor2tensor.utils import decoding
from tensor2tensor.utils import trainer_lib

from magenta.models.score2perf import score2perf
import note_seq
from tensorflow import ConfigProto
from tensorflow import InteractiveSession
"""
ランダムな曲を生成するスクリプト
"""
config = ConfigProto()
config.gpu_options.allow_growth = False
session = InteractiveSession(config=config)
tf.disable_v2_behavior()

# モデルへ入力を渡す関数（generator）
targets = []
decode_length = 0
def input_generator():
    global targets
    global decode_length
    while True:
        yield {
            "targets": np.array([targets], dtype=np.int32),
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
# ckpt_path = "/mnt/c/Users/hayat/Desktop/myself/models/transfomer/melody_conditioned_model_16.ckpt"  # チェックポイント
ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/unconditional_model_16.ckpt'
# ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/melody_conditioned_model_16.ckpt'
# エンコーダー生成用のクラス
class PianoPerformanceProblem(score2perf.Score2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = PianoPerformanceProblem()
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

# # # 最初の推定結果は飛ばす
next(predicted)

targets = []
decode_length = 1024

# 推定結果をidとして取得
predicted_ids = next(predicted)["outputs"]

# idをNoteSequenceに変換
midi_file = decode(
    predicted_ids,
    encoder=encoders["targets"]
    )
seq = note_seq.midi_file_to_note_sequence(midi_file)

# 再生と楽譜の表示
# note_seq.plot_sequence(seq)
# note_seq.play_sequence(seq, synth=note_seq.fluidsynth) 

note_seq.sequence_proto_to_midi_file(seq, "music_transformer_composition_conditional.mid")  #MIDI　データに変換し保存
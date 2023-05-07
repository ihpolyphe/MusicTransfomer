

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
入力midに対して続きを作曲するスクリプト
!伴奏用のモデルのため使用してもいい感じのモデルは作れないよ
"""
tf.disable_v2_behavior()
# 出だしをインプット
input_midi_path = "/mnt/c/Users/hayat/Desktop/myself/input_code/yorunikakeru/yorunikakeru_short.mid"
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
# ckpt_path = 'gs://magentadata/models/music_transformer/checkpoints/melody_conditioned_model_16.ckpt'
ckpt_path = "/mnt/c/Users/hayat/Desktop/myself/models/transfomer/melody_conditioned_model_16.ckpt"  # チェックポイント

# エンコーダー生成用のクラス
class MelodyToPianoPerformanceProblem(score2perf.AbsoluteMelody2PerfProblem):
  @property
  def add_eos_symbol(self):
    return True

problem = MelodyToPianoPerformanceProblem()
conditional_encoders = problem.get_feature_encoders()

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

# targets = []
# decode_length = 1024

# # 推定結果をidとして取得
# predicted_ids = next(predicted)["outputs"]

# # idをNoteSequenceに変換
# midi_file = decode(
#     predicted_ids,
#     encoder=unconditional_encoders["targets"]
#     )
# unconditional_ns = note_seq.midi_file_to_note_sequence(midi_file)
# note_seq.sequence_proto_to_midi_file(unconditional_ns, "music_transformer_composition_conditional.mid")  #MIDI　データに変換し保存



input_midi = note_seq.midi_file_to_note_sequence(input_midi_path)
# Handle sustain pedal in the primer.
primer_ns = note_seq.apply_sustain_control_changes(input_midi)

# Trim to desired number of seconds.
max_primer_seconds = 20  #@param {type:"slider", min:1, max:120}
if primer_ns.total_time > max_primer_seconds:
  print('Primer is longer than %d seconds, truncating.' % max_primer_seconds)
  primer_ns = note_seq.extract_subsequence(
      primer_ns, 0, max_primer_seconds)

# Remove drums from primer if present.
if any(note.is_drum for note in primer_ns.notes):
  print('Primer contains drums; they will be removed.')
  notes = [note for note in primer_ns.notes if not note.is_drum]
  del primer_ns.notes[:]
  primer_ns.notes.extend(notes)

# Set primer instrument and program.
for note in primer_ns.notes:
  note.instrument = 1
  note.program = 0

#@title Generate Continuation
#@markdown Continue a piano performance, starting with the
#@markdown chosen priming sequence.

targets = conditional_encoders['targets'].encode_note_sequence(
    primer_ns)

# Remove the end token from the encoded primer.
targets = targets[:-1]

decode_length = max(0, 4096 - len(targets))
if len(targets) >= 4096:
  print('Primer has more events than maximum sequence length; nothing will be generated.')

# Generate sample events.
sample_ids = next(predicted)['outputs']

# Decode to NoteSequence.
midi_filename = decode(
    sample_ids,
    encoder=conditional_encoders['targets'])
ns = note_seq.midi_file_to_note_sequence(midi_filename)

# Append continuation to primer.
continuation_ns = note_seq.concatenate_sequences([primer_ns, ns])
now = datetime.datetime.now()
filename = "/mnt/c/Users/hayat/Desktop/myself/music_transfomer/result/conditional_generate_" + now.strftime('%Y_%m_%d_%H_%M_%S') + '.mid'
note_seq.sequence_proto_to_midi_file(continuation_ns, filename)
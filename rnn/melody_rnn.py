import magenta
import note_seq
from note_seq.protobuf import music_pb2, generator_pb2

from magenta.models.melody_rnn import melody_rnn_sequence_generator
from magenta.models.shared import sequence_generator_bundle

"""
きらきら星の続きをRNNで作成するscript
"""


kira2 = music_pb2.NoteSequence()  # NoteSequence

# notesにnoteを追加
kira2.notes.add(pitch=60, start_time=0.0, end_time=0.4, velocity=80)
kira2.notes.add(pitch=60, start_time=0.4, end_time=0.8, velocity=80)
kira2.notes.add(pitch=67, start_time=0.8, end_time=1.2, velocity=80)
kira2.notes.add(pitch=67, start_time=1.2, end_time=1.6, velocity=80)
kira2.notes.add(pitch=69, start_time=1.6, end_time=2.0, velocity=80)
kira2.notes.add(pitch=69, start_time=2.0, end_time=2.4, velocity=80)
kira2.notes.add(pitch=67, start_time=2.4, end_time=3.2, velocity=80)
kira2.notes.add(pitch=65, start_time=3.2, end_time=3.6, velocity=80)
kira2.notes.add(pitch=65, start_time=3.6, end_time=4.0, velocity=80)
kira2.notes.add(pitch=64, start_time=4.0, end_time=4.4, velocity=80)
kira2.notes.add(pitch=64, start_time=4.4, end_time=4.8, velocity=80)
kira2.notes.add(pitch=62, start_time=4.8, end_time=5.2, velocity=80)
kira2.notes.add(pitch=62, start_time=5.2, end_time=5.6, velocity=80)
kira2.notes.add(pitch=60, start_time=5.6, end_time=6.4, velocity=80) 

kira2.total_time = 6.4  # 所要時間
kira2.tempos.add(qpm=75);  # 曲のテンポを指定

# note_seq.plot_sequence(kira2)  # NoteSequenceの可視化
# note_seq.play_sequence(kira2, synth=note_seq.fluidsynth)  # NoteSequenceの再生

# note_seq.sequence_proto_to_midi_file(kira2, "kira2_second.mid")  #MIDI　データに変換し保存

# モデルの初期化
# note_seq.notebook_utils.download_bundle("basic_rnn.mag", "/models/")  # Bundle（.magファイル）をダウンロード
bundle = sequence_generator_bundle.read_bundle_file("/mnt/c/Users/hayat/Desktop/myself/models/basic_rnn.mag")  # Bundleの読み込み
generator_map = melody_rnn_sequence_generator.get_generator_map()
melody_rnn = generator_map["basic_rnn"](checkpoint=None, bundle=bundle)  # 生成器の設定
melody_rnn.initialize()  # 初期化

# generate music
base_sequence = kira2  # ベースになるNoteSeqence
total_time = 36 # 曲の長さ（秒）
temperature = 1.2 # 曲の「ランダム度合い」を決める定数

base_end_time = max(note.end_time for note in base_sequence.notes)  #ベース曲の終了時刻

# 生成器に関する設定
generator_options = generator_pb2.GeneratorOptions()  # 生成器のオプション
generator_options.args["temperature"].float_value = temperature  # ランダム度合い
generator_options.generate_sections.add(
    start_time=base_end_time,  # 作曲開始時刻
    end_time=total_time)  # 作曲終了時刻

# 曲の生成
gen_seq = melody_rnn.generate(base_sequence, generator_options)

# note_seq.plot_sequence(gen_seq)  # NoteSequenceの可視化
# note_seq.play_sequence(gen_seq, synth=note_seq.fluidsynth)  # NoteSequenceの再生
     
# save mid file
note_seq.sequence_proto_to_midi_file(gen_seq, "simple_melody_rnn.mid")  #MIDI　データに変換し保存
# files.download("simple_melody_rnn.mid")  # ダウンロード
     
import magenta
import note_seq
from note_seq.protobuf import music_pb2
from magenta.models.pianoroll_rnn_nade import pianoroll_rnn_nade_sequence_generator
from magenta.models.shared import sequence_generator_bundle
from note_seq.protobuf import generator_pb2

seed = music_pb2.NoteSequence()  # NoteSequence

# notesにnoteを追加
seed.notes.add(pitch=80, start_time=0.0, end_time=0.4, velocity=80)
seed.notes.add(pitch=80, start_time=0.4, end_time=0.8, velocity=80)
seed.notes.add(pitch=87, start_time=0.8, end_time=1.2, velocity=80)
seed.notes.add(pitch=87, start_time=1.2, end_time=1.6, velocity=80)
seed.notes.add(pitch=89, start_time=1.6, end_time=2.0, velocity=80)
seed.notes.add(pitch=89, start_time=2.0, end_time=2.4, velocity=80)

seed.notes.add(pitch=60, start_time=0.0, end_time=0.4, velocity=80)
seed.notes.add(pitch=62, start_time=0.4, end_time=0.8, velocity=80)
seed.notes.add(pitch=64, start_time=0.8, end_time=1.2, velocity=80)
seed.notes.add(pitch=67, start_time=1.2, end_time=1.6, velocity=80)
seed.notes.add(pitch=69, start_time=1.6, end_time=2.0, velocity=80)
seed.notes.add(pitch=64, start_time=2.0, end_time=2.4, velocity=80)

seed.notes.add(pitch=49, start_time=0.0, end_time=0.8, velocity=80)
seed.notes.add(pitch=44, start_time=0.8, end_time=1.6, velocity=80)
seed.notes.add(pitch=40, start_time=1.6, end_time=2.4, velocity=80)

seed.total_time = 2.4  # 所要時間
seed.tempos.add(qpm=75);  # 曲のテンポを指定



# モデルの初期化
# note_seq.notebook_utils.download_bundle("pianoroll_rnn_nade.mag", "/models/")  # Bundle（.magファイル）をダウンロード
bundle = sequence_generator_bundle.read_bundle_file("/mnt/c/Users/hayat/Desktop/myself/models/pianoroll_rnn_nade.mag")  # Bundleの読み込み
generator_map = pianoroll_rnn_nade_sequence_generator.get_generator_map()
pianoroll_rnn_nade = generator_map["rnn-nade_attn"](checkpoint=None, bundle=bundle)  # 生成器の設定
pianoroll_rnn_nade.initialize()  # 初期化


total_time = 180 # 曲の長さ（秒）
temperature = 1.0 # 曲の「ランダム度合い」を決める定数

base_end_time = max(note.end_time for note in seed.notes)  #ベース曲の終了時刻

# 生成器に関する設定
generator_options = generator_pb2.GeneratorOptions()  # 生成器のオプション
generator_options.args["temperature"].float_value = temperature  # ランダム度合い
generator_options.generate_sections.add(
    start_time=base_end_time,  # 作曲開始時刻
    end_time=total_time)  # 作曲終了時刻

# 曲の生成
gen_seq = pianoroll_rnn_nade.generate(seed, generator_options)
note_seq.sequence_proto_to_midi_file(gen_seq, "pianoroll_rnn_nade.mid")  #MIDI　データに変換し保存
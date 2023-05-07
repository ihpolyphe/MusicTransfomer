from magenta.models.music_vae import configs
from magenta.models.music_vae.trained_model import TrainedModel

import note_seq
import magenta
from note_seq.protobuf import music_pb2

# モデルの初期化
music_vae = TrainedModel(
      configs.CONFIG_MAP["cat-mel_2bar_big"], 
      batch_size=4,  # 一度に処理するデータ数
      checkpoint_dir_or_path="/mnt/c/Users/hayat/Desktop/myself/models/mel_2bar_big.ckpt")



generated = music_vae.sample(n=5,  # 生成数
                             length=128,  # ステップ数
                             temperature=1.0)  # 温度

# 最初のNoteSeqence
kira2_start = music_pb2.NoteSequence()

kira2_start.notes.add(pitch=60, start_time=0.0, end_time=0.4, velocity=80)
kira2_start.notes.add(pitch=60, start_time=0.4, end_time=0.8, velocity=80)
kira2_start.notes.add(pitch=67, start_time=0.8, end_time=1.2, velocity=80)
kira2_start.notes.add(pitch=67, start_time=1.2, end_time=1.6, velocity=80)
kira2_start.notes.add(pitch=69, start_time=1.6, end_time=2.0, velocity=80)
kira2_start.notes.add(pitch=69, start_time=2.0, end_time=2.4, velocity=80)
kira2_start.notes.add(pitch=67, start_time=2.4, end_time=3.2, velocity=80)
kira2_start.notes.add(pitch=65, start_time=3.2, end_time=3.6, velocity=80)
kira2_start.notes.add(pitch=65, start_time=3.6, end_time=4.0, velocity=80)
kira2_start.notes.add(pitch=64, start_time=4.0, end_time=4.4, velocity=80)
kira2_start.notes.add(pitch=64, start_time=4.4, end_time=4.8, velocity=80)
kira2_start.notes.add(pitch=62, start_time=4.8, end_time=5.2, velocity=80)
kira2_start.notes.add(pitch=62, start_time=5.2, end_time=5.6, velocity=80)
kira2_start.notes.add(pitch=60, start_time=5.6, end_time=6.4, velocity=80) 

kira2_start.total_time = 6.4 
kira2_start.tempos.add(qpm=75);

note_seq.plot_sequence(kira2_start)
note_seq.play_sequence(kira2_start, synth=note_seq.fluidsynth)

# 最後のNoteSeqence
kira2_end = music_pb2.NoteSequence()

kira2_end.notes.add(pitch=70, start_time=0.0, end_time=0.4, velocity=80)
kira2_end.notes.add(pitch=72, start_time=0.4, end_time=0.8, velocity=80)
kira2_end.notes.add(pitch=74, start_time=0.8, end_time=1.2, velocity=80)
kira2_end.notes.add(pitch=77, start_time=1.2, end_time=1.6, velocity=80)
kira2_end.notes.add(pitch=79, start_time=1.6, end_time=2.0, velocity=80)
kira2_end.notes.add(pitch=74, start_time=2.0, end_time=2.4, velocity=80)
kira2_end.notes.add(pitch=70, start_time=2.4, end_time=3.2, velocity=80)
kira2_end.notes.add(pitch=72, start_time=3.2, end_time=3.6, velocity=80)
kira2_end.notes.add(pitch=74, start_time=3.6, end_time=4.0, velocity=80)
kira2_end.notes.add(pitch=77, start_time=4.0, end_time=4.4, velocity=80)
kira2_end.notes.add(pitch=79, start_time=4.4, end_time=4.8, velocity=80)
kira2_end.notes.add(pitch=74, start_time=4.8, end_time=5.2, velocity=80)
kira2_end.notes.add(pitch=72, start_time=5.2, end_time=5.6, velocity=80)
kira2_end.notes.add(pitch=70, start_time=5.6, end_time=6.4, velocity=80) 

kira2_end.total_time = 6.4
kira2_end.tempos.add(qpm=75); 

n_seq = 8  # 曲のNoteSeqence数（最初と最後を含む）

# NoteSeqenceを複数生成し、リストに格納
gen_seq = music_vae.interpolate(
    kira2_start,  # 最初のNoteSeqence
    kira2_end,  # 最後のNoteSeqence
    num_steps=n_seq,
    length=32)

# NoteSeqenceを全て結合し、1つの曲に
interp_seq = note_seq.sequences_lib.concatenate_sequences(gen_seq)
note_seq.sequence_proto_to_midi_file(interp_seq, "simple_music_vae.mid")  #MIDI　データに変換し保存
import os
import glob
import torch
import torchaudio
import librosa
import soundfile as sf
from rich.progress import track
from joblib import Parallel, delayed


def get_sil_indexes(wav_path, sr, trim_db=30):
    wav, _ = librosa.load(wav_path, sr=sr)
    _, [start, end] = librosa.effects.trim(wav, top_db=trim_db)
    start = start - min(start, int(0.1 * sr))
    end = end + min(len(wav) - end, int(0.1 * sr))
    # wav_trimmed = wav[start: end]
    return [start, end]


def trim_silence(raw_wav_path, out_wav_path, sr, sil_index):
    os.makedirs(os.path.dirname(out_wav_path), exist_ok=True)

    start, end = sil_index[0], sil_index[1]
    raw_wav, _ = librosa.load(raw_wav_path, sr=sr)
    trimmed_wav = raw_wav[start: end]
    sf.write(out_wav_path, trimmed_wav, sr, 'PCM_16')


def main(raw_path, out_path, sampling_rate):
    cln_index = 'clean'

    env_indexes = ['confroom1', 'confroom2', 'livingroom1', 'office1', 'office2', 'studio1', 'studio2', 'studio3', 'waitingroom1']

    wav_indexes = [x.split('.')[0] for x in sorted(os.listdir(os.path.join(raw_path, cln_index)))]

    silence_indexes = {}
    print('Extracting Audio Silence Indexes and Trimming Clean Audios.')
    os.makedirs(os.path.join(out_path, cln_index), exist_ok=True)
    for wav_index in track(wav_indexes):
        silence_indexes[wav_index] = get_sil_indexes(os.path.join(raw_path, cln_index, wav_index+'.flac'), sampling_rate)
        trim_silence(os.path.join(raw_path, cln_index, wav_index+'.flac'), 
                     os.path.join(out_path, cln_index, wav_index+'.wav'), 
                     sampling_rate, 
                     silence_indexes[wav_index])

    for env_index in env_indexes:
        print('Trimming Audios of ' + env_index + '.')
        env_wav_paths = glob.glob(os.path.join(raw_path, env_index, '*/*/*.flac'))

        # VCTK
        # Parallel(n_jobs=32)(delayed(trim_silence)(
        #     env_wav_path, 
        #     env_wav_path.replace(raw_path, out_path).replace('.flac', '.wav'), 
        #     sampling_rate, 
        #     silence_indexes['_'.join(env_wav_path.split(os.sep)[-1].split('_')[:2])]) for env_wav_path in track(env_wav_paths))
        
        # DASP
        Parallel(n_jobs=32)(delayed(trim_silence)(
            env_wav_path, 
            env_wav_path.replace(raw_path, out_path).replace('.flac', '.wav'), 
            sampling_rate, 
            silence_indexes['_'.join(env_wav_path.split(os.sep)[-1].split('_')[:4])]) for env_wav_path in track(env_wav_paths))


if __name__=='__main__':

    # raw_path = 'DDS/VCTK_16k'
    # out_path = 'DDS/VCTK_16k_trimmed'

    raw_path = 'DDS/DAPS_16k'
    out_path = 'DDS/DAPS_16k_trimmed'

    sampling_rate = 16000
    main(raw_path, out_path, sampling_rate)

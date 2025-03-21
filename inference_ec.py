import os
import argparse
import torch
import torchaudio
import utils
from models.model import SynthesizerTrn
from speaker_encoder.speakers import SpeakerManager
from text.symbols import symbols
from datasets.mel_processing import spectrogram_torch
from tqdm import tqdm

hps = None
device = None
net_g = None
spk_enc = None


def environment_conversion_sample(src_wav_path, tgt_wav_path, out_wav_path):
    
    audio_src, sr = torchaudio.load(src_wav_path)
    audio_tgt, sr = torchaudio.load(tgt_wav_path)
    assert sr == hps.data.sampling_rate

    spec_src = spectrogram_torch(audio_src, hps.data.filter_length, hps.data.hop_length, hps.data.win_length).to(device)
    spec_tgt = spectrogram_torch(audio_tgt, hps.data.filter_length, hps.data.hop_length, hps.data.win_length).to(device)
    spec_src_lengths = torch.LongTensor([spec_src.size(2)]).to(device)
    spec_tgt_lengths = torch.LongTensor([spec_tgt.size(2)]).to(device)

    spk_embedd = spk_enc.compute_embedding_from_clip(src_wav_path)
    spk_embedd = torch.FloatTensor(spk_embedd).unsqueeze(0).to(device)

    audio_env = net_g.environment_conversion(spec_src, spec_tgt, spec_src_lengths, spec_tgt_lengths, spk_embedd)[0][0,0].data.unsqueeze(0).cpu()
    torchaudio.save(out_wav_path, audio_env, hps.data.sampling_rate, encoding='PCM_S', bits_per_sample=16)


def inference(a):
    with open(a.input_text_file, 'r', encoding='utf-8') as fi:
        _, audio_paths_src, _, audio_paths_tgt = zip(*(line.strip().split('|') for line in fi.read().split('\n') if len(line) > 0))
    
    os.makedirs(a.output_wavs_dir, exist_ok=True)

    with torch.no_grad():
        for i in tqdm(range(len(audio_paths_src))):
            env_wav_path = os.path.join(a.input_wavs_dir, audio_paths_src[i]+'.flac')
            tgt_wav_path = os.path.join(a.input_wavs_dir, audio_paths_tgt[i]+'.flac')
            out_wav_path = os.path.join(a.output_wavs_dir, audio_paths_src[i].split(os.sep)[-1].replace('_livingroom1_Uber_ch1', '')+'.wav')
            environment_conversion_sample(env_wav_path, tgt_wav_path, out_wav_path)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--se_ckpt_path', default='checkpoints/ckpt_se/model_se.pth.tar')
    parser.add_argument('--se_conf_path', default='checkpoints/ckpt_se/config_se.json')
    parser.add_argument('--input_text_file', default='../filelists_ec/env2clean.txt')
    parser.add_argument('--input_wavs_dir', default='/disk1/yxlu/datasets/DDS')
    parser.add_argument('--output_wavs_dir', default='../generated_files')
    parser.add_argument('--checkpoint_file', required=True)
    a = parser.parse_args()

    global hps
    config_path = os.path.join(os.path.dirname(a.checkpoint_file), 'config.json')
    hps = utils.get_hparams_from_file(config_path)

    use_cuda = torch.cuda.is_available()
    global device
    if use_cuda:
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    global net_g
    net_g = SynthesizerTrn(
    len(symbols),
    hps.data.filter_length // 2 + 1,
    hps.train.segment_size // hps.data.hop_length,
    n_speakers=hps.data.n_speakers,
    **hps.model).to(device)

    _ = net_g.eval()
    _ = utils.load_checkpoint(a.checkpoint_file, net_g, None)

    global spk_enc
    
    spk_enc = SpeakerManager(encoder_model_path=a.se_ckpt_path, encoder_config_path=a.se_conf_path, use_cuda=use_cuda)

    inference(a)

if __name__ == '__main__':
    main()

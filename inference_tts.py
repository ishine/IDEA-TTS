import os
import argparse
import torch
import torchaudio
from models import commons
import utils
from models.model import SynthesizerTrn
from speaker_encoder.speakers import SpeakerManager
from text.symbols import symbols
from text import text_to_sequence
from datasets.mel_processing import spectrogram_torch
from tqdm import tqdm


hps = None
device = None
net_g = None
spk_enc = None


def get_text(text, hps):
    text_norm = text_to_sequence(text, hps.data.text_cleaners)
    if hps.data.add_blank:
        text_norm = commons.intersperse(text_norm, 0)
    text_norm = torch.LongTensor(text_norm)
    return text_norm


def inference_tts_sample(text, ref_spk_path, ref_env_path, out_cln_path, out_env_path):
    stn_tst = get_text(text, hps)
    x_tst = stn_tst.unsqueeze(0).to(device)
    x_tst_lengths = torch.LongTensor([stn_tst.size(0)]).to(device)

    spk_embedd = spk_enc.compute_embedding_from_clip(ref_spk_path)
    spk_embedd = torch.FloatTensor(spk_embedd).unsqueeze(0).to(device)

    ref_env_audio, sr = torchaudio.load(ref_env_path)
    assert sr == hps.data.sampling_rate

    spec_env = spectrogram_torch(ref_env_audio, hps.data.filter_length, hps.data.hop_length, hps.data.win_length).to(device)
    spec_env_lengths = torch.LongTensor([spec_env.size(2)]).to(device)

    audio_g_cln, audio_g_env, *_ = net_g.infer(x_tst, x_tst_lengths, spec_env, spec_env_lengths, spk_embedd, noise_scale=.667, noise_scale_w=0.8, length_scale=1)

    torchaudio.save(out_cln_path, audio_g_cln[0,0].data.unsqueeze(0).cpu(), hps.data.sampling_rate, encoding='PCM_S', bits_per_sample=16)
    torchaudio.save(out_env_path, audio_g_env[0,0].data.unsqueeze(0).cpu(), hps.data.sampling_rate, encoding='PCM_S', bits_per_sample=16)


def inference(a):
    with open(a.input_text_file, 'r', encoding='utf-8') as fi:
        ref_txt_indexes, texts, ref_spk_indexes, ref_env_indexes = zip(*(line.strip().split('|') for line in fi.read().split('\n') if len(line) > 0))
    
    output_wavs_dir_cln = os.path.join(a.output_wavs_dir, 'clean')
    output_wavs_dir_env = os.path.join(a.output_wavs_dir, 'env')
    os.makedirs(output_wavs_dir_cln, exist_ok=True)
    os.makedirs(output_wavs_dir_env, exist_ok=True)

    with torch.no_grad():
        for i, text in enumerate(tqdm(texts)):
            ref_spk_path = os.path.join(a.input_wavs_dir, ref_spk_indexes[i])
            ref_env_path = os.path.join(a.input_wavs_dir, ref_env_indexes[i])

            out_cln_path = os.path.join(output_wavs_dir_cln, ref_txt_indexes[i].split(os.sep)[-1].split('.')[0] + '|' + ref_spk_indexes[i].split(os.sep)[-1].split('.')[0] + '|' + ref_env_indexes[i].split(os.sep)[-1])
            out_env_path = os.path.join(output_wavs_dir_env, ref_txt_indexes[i].split(os.sep)[-1].split('.')[0] + '|' + ref_spk_indexes[i].split(os.sep)[-1].split('.')[0] + '|' + ref_env_indexes[i].split(os.sep)[-1])

            inference_tts_sample(text, ref_spk_path, ref_env_path, out_cln_path, out_env_path)


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--se_ckpt_path', default='speaker_encoder/checkpoints/model_se.pth.tar')
    parser.add_argument('--se_conf_path', default='speaker_encoder/checkpoints/config_se.json')
    parser.add_argument('--input_text_file', default='filelists/tts_test.txt')
    parser.add_argument('--input_wavs_dir', default='/home/yxlu/datasets/DDS')
    parser.add_argument('--output_wavs_dir', default='generated_files')
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
import os
import glob
import torch
from speaker_encoder.speakers import SpeakerManager
from tqdm import tqdm

### Speaker Encoder Initializing
SPEAKER_ENCODER_CHECKPOINT_PATH = ("checkpoints/ckpt_se/model_se.pth.tar")
SPEAKER_ENCODER_CONFIG_PATH = "checkpoints/ckpt_se/config_se.json"
USE_CUDA = torch.cuda.is_available()
SE_speaker_manager = SpeakerManager(encoder_model_path=SPEAKER_ENCODER_CHECKPOINT_PATH, encoder_config_path=SPEAKER_ENCODER_CONFIG_PATH, use_cuda=USE_CUDA)


environments = ["confroom1", "confroom2", "livingroom1", "office1", "office2", "studio1", "studio2", "studio3", "waitingroom1"]
devices = ["iPad", "Marantz", "Uber"]
positions = ["ch1", "ch2", "ch3", "ch4", "ch5", "ch6"]

dataset_path = 'DDS/DAPS_16k_trimmed'
embedding_path = 'DDS/DAPS_16k_trimmed_embedding'

cln_wav_indexes = os.listdir(os.path.join(dataset_path, 'clean'))
os.makedirs(os.path.join(embedding_path, 'clean'), exist_ok=True)
print("Processing Clean.")
for cln_wav_index in tqdm(cln_wav_indexes):
    embedd = SE_speaker_manager.compute_embedding_from_clip(os.path.join(dataset_path, 'clean', cln_wav_index))
    embedd = torch.FloatTensor(embedd)
    torch.save(embedd, os.path.join(embedding_path, 'clean', cln_wav_index.replace('.wav', '.pt')))

for env in environments:
    for device in devices:
        for position in positions:
            env_wav_indexes = os.listdir(os.path.join(dataset_path, env, device, position))
            os.makedirs(os.path.join(embedding_path, env, device, position), exist_ok=True)
            print("Processing {} {} {}.".format(env, device, position))
            for env_wav_index in tqdm(env_wav_indexes):
                embedd = SE_speaker_manager.compute_embedding_from_clip(os.path.join(dataset_path, env, device, position, env_wav_index))
                embedd = torch.FloatTensor(embedd)
                torch.save(embedd, os.path.join(embedding_path, env, device, position, env_wav_index.replace('.wav', '.pt')))

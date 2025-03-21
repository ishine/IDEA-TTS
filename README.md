# Incremental Disentanglement for Environment-Aware Zero-Shot Text-to-Speech Synthesis
### Ye-Xin Lu, Hui-Peng Du, Zheng-Yan Sheng, Yang Ai, Zhen-Hua Ling

**Abstract:** 
This paper proposes IDEA-TTS, an Incremental Disentanglement-based Environment-Aware zero-shot text-to-speech (TTS) method that can synthesize speech for unseen speakers while preserving the acoustic characteristics of a given environment reference speech. IDEA-TTS adopts VITS as the TTS backbone. To effectively disentangle the environment, speaker, and text factors, we propose an incremental disentanglement process, where an environment estimator is designed to first decompose the environmental spectrogram into an environment mask and an enhanced spectrogram. The environment mask is then processed by an environment encoder to extract environment embeddings, while the enhanced spectrogram facilitates the subsequent disentanglement of the speaker and text factors with the condition of the speaker embeddings, which are extracted from the environmental speech using a pretrained environment-robust speaker encoder. Finally, both the speaker and environment embeddings are conditioned into the decoder for environment-aware speech generation. Experimental results demonstrate that IDEA-TTS achieves superior performance in the environment-aware TTS task, excelling in speech quality, speaker similarity, and environmental similarity. Additionally, IDEA-TTS is also capable of the acoustic environment conversion task and achieves state-of-the-art performance.

**We provide our implementation as open source in this repository. Audio samples can be found at the  [demo website](http://yxlu-0102.github.io/IDEA-TTS).**

## Pre-requisites
1. Clone this repository
2. Install python requirements.
   1. You may need first to install the requirements of the [YourTTS](https://github.com/coqui-ai/TTS) to extract speaker embeddings by:
        ```
        $ conda create -n yourtts python=3.9
        $ git clone https://github.com/coqui-ai/TTS
        $ pip install -e .
        ```
    2. Then install the requirements of the IDEA-TTS referring [requirements.txt](requirements.txt):
        ```
        $ conda create -n ideatts python=3.9
        $ pip install -r requirements.txt
        ```
3. Download datasets
    1. Download and extract the [DDS dataset](https://zenodo.org/records/5464104).
    2. Trim the silence of the dataset, and the trimmed files will be saved to `DDS/VCTK_16k_trimmed` and `DDS/DAPS_16k_trimmed`.
       ```
       $ python trim_silence.py
       ```
4. Extract speaker embeddings:
   1. Download the [checkpoint files](https://drive.google.com/drive/folders/1hGdJFUOwSrN8ClieUSvfIXyxIcilOMCo?usp=share_link) of the speaker encoder and pretrained IDEA-TTS, and move them to the `checkpoint` dir.
   2. Preprocess the dataset to extract speaker embeddings
       ```
       $ conda activate yourtts
       $ python preprocess_se.py
       ```
5. Build Monotonic Alignment Search
   ```
   cd monotonic_align
   python setup.py build_ext --inplace
   ```
## Training
```
$ conda activate ideatts
$ CUDA_VISIBLE_DEVICES=0 python train.py -c [config file path] -m [checkpoint file path]
```
The checkpoint file will be saved in the `[checkpoint file path]`, here's an example:
```
$ CUDA_VISIBLE_DEVICES=0 python train.py -c config.json -m checkpoints/IDEA-TTS
```
## Inference
1. Inference for the environment-aware TTS task
   ```
   $ conda activate yourtts
   $ CUDA_VISIBLE_DEVICES=0 python inference_tts.py --checkpoint_file [checkpoint file path] --output_wavs_dir [output dir path]
   ```
   You can use the pretrained weights we provide. Generated wav files are saved in the `[output dir path]`. Here is an example:
   ```
   $ CUDA_VISIBLE_DEVICES=0 python inference_tts.py --checkpoint_file checkpoints/ckpt_tts/model.pth  --output_wavs_dir generated_files/EA-TTS
   ```
2. Inference for the acoustic environment conversion task
   ```
   $ conda activate yourtts
   $ CUDA_VISIBLE_DEVICES=0 python inference_ec.py -checkpoint_file [checkpoint file path] --output_wavs_dir [output dir path]
   ```
   Here is an example:
   ```
   $ CUDA_VISIBLE_DEVICES=0 python inference_ec.py --checkpoint_file checkpoints/ckpt_tts/model.pth --input_text_file filelists_ec/env2clean.txt --output_wavs_dir generated_files/Env2Clean
   ```
## Acknowledgements
We referred to [VITS](https://github.com/jaywalnut310/vits) and [YourTTS](https://github.com/coqui-ai/TTS) to implement this.

## Citation
```
@inproceedings{lu2025incremental,
  title={Incremental Disentanglement for Environment-Aware Zero-Shot Text-to-Speech Synthesis},
  author={Lu, Ye-Xin and Du, Hui-Peng and Sheng, Zheng-Yan and Ai, Yang and Ling, Zhen-Hua},
  booktitle={Proc. ICASSP},
  year={2025}
}
```

import os
import random
import torch
import torchaudio
import torchaudio.functional as aF
import torch.utils.data

import models.commons as commons
from datasets.mel_processing import spectrogram_torch
from utils import load_filepaths_and_text, get_env_path
from text import text_to_sequence, cleaned_text_to_sequence


"""Multi speaker version"""
class TextAudioSpeakerLoader(torch.utils.data.Dataset):
    """
        1) loads audio, speaker_id, text pairs
        2) normalizes text and converts them to sequences of integers
        3) computes spectrograms from audio files.
    """
    def __init__(self, audioindex_text, hparams, fix_env=False):
        self.audio_path = hparams.audio_path
        self.audioindex_text = load_filepaths_and_text(audioindex_text)
        self.environments = hparams.environments
        self.devices = hparams.devices
        self.positions = hparams.positions
        self.text_cleaners = hparams.text_cleaners
        self.sampling_rate = hparams.sampling_rate
        self.filter_length = hparams.filter_length
        self.hop_length    = hparams.hop_length
        self.win_length    = hparams.win_length
        self.sampling_rate = hparams.sampling_rate

        self.cleaned_text = getattr(hparams, "cleaned_text", False)

        self.add_blank = hparams.add_blank
        self.min_text_len = getattr(hparams, "min_text_len", 1)
        self.max_text_len = getattr(hparams, "max_text_len", 190)

        self.fix_env = fix_env

        random.seed(1234)
        random.shuffle(self.audioindex_text)
        self._filter()

    def _filter(self):
        """
        Filter text & store spec lengths
        """

        audioindex_text_new = []
        lengths = []
        for audioindex, text in self.audioindex_text:
            if self.min_text_len <= len(text) and len(text) <= self.max_text_len:
                audioindex_text_new.append([audioindex, text])
                lengths.append(os.path.getsize(os.path.join(self.audio_path, audioindex)) // (2 * self.hop_length))
        self.audioindex_text = audioindex_text_new
        self.lengths = lengths

    def get_audio_text_pair(self, audioindex_text):
        # separate filename, speaker_id and text
        audioindex, text = audioindex_text[0], audioindex_text[1]
        text = self.get_text(text)
        cln_spec, cln_wav, cln_embedd = self.get_audio(audioindex)
        if self.fix_env:
            env_spec, env_wav, env_embedd = self.get_audio(get_env_path(audioindex, 'livingroom1', 'Uber', 'ch1', self.fix_env))
        else:
            env_spec, env_wav, env_embedd = self.get_audio(get_env_path(audioindex, 
                                                            random.choice(self.environments), 
                                                            random.choice(self.devices), 
                                                            random.choice(self.positions),
                                                            self.fix_env))
        return (text, cln_spec, cln_wav, cln_embedd, env_spec, env_wav, env_embedd)

    def get_audio(self, filename):
        audio, sampling_rate = torchaudio.load(os.path.join(self.audio_path, filename))
        if sampling_rate != self.sampling_rate:
            print(filename)
            raise ValueError("{} SR doesn't match target {} SR".format(sampling_rate, self.sampling_rate))

        spec = spectrogram_torch(audio, self.filter_length, self.hop_length, self.win_length, center=False)
        spec = torch.squeeze(spec, 0)

        embedd = torch.load(os.path.join(self.audio_path, filename.replace('trimmed', 'trimmed_embedding').replace('.wav', '.pt')))

        return spec, audio, embedd

    def get_text(self, text):
        if self.cleaned_text:
            text_norm = cleaned_text_to_sequence(text)
        else:
            text_norm = text_to_sequence(text, self.text_cleaners)
        if self.add_blank:
            text_norm = commons.intersperse(text_norm, 0)
        text_norm = torch.LongTensor(text_norm)
        return text_norm

    def __getitem__(self, index):
        return self.get_audio_text_pair(self.audioindex_text[index])

    def __len__(self):
        return len(self.audioindex_text)


class TextAudioSpeakerCollate():
    """ Zero-pads model inputs and targets
    """
    def __init__(self, return_ids=False):
        self.return_ids = return_ids

    def __call__(self, batch):
        """Collate's training batch from normalized text, audio and speaker identities
        PARAMS
        ------
        batch: text, cln_spec, cln_wav, cln_embedd, env_spec, env_wav, env_embedd
        """
        # Right zero-pad all one-hot text sequences to max input length
        _, ids_sorted_decreasing = torch.sort(
            torch.LongTensor([x[1].size(1) for x in batch]),
            dim=0, descending=True)

        max_text_len = max([len(x[0]) for x in batch])
        max_spec_len = max([x[1].size(1) for x in batch])
        max_wav_len = max([x[2].size(1) for x in batch])

        text_lengths = torch.LongTensor(len(batch))
        spec_lengths = torch.LongTensor(len(batch))
        wav_lengths = torch.LongTensor(len(batch))

        text_padded = torch.LongTensor(len(batch), max_text_len)
        cln_spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        cln_wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        cln_embedd = torch.FloatTensor(len(batch), batch[0][3].size(0))
        env_spec_padded = torch.FloatTensor(len(batch), batch[0][1].size(0), max_spec_len)
        env_wav_padded = torch.FloatTensor(len(batch), 1, max_wav_len)
        env_embedd = torch.FloatTensor(len(batch), batch[0][3].size(0))

        text_padded.zero_()
        cln_spec_padded.zero_()
        cln_wav_padded.zero_()
        cln_embedd.zero_()
        env_spec_padded.zero_()
        env_wav_padded.zero_()
        env_embedd.zero_()

        for i in range(len(ids_sorted_decreasing)):
            row = batch[ids_sorted_decreasing[i]]

            text = row[0]
            text_padded[i, :text.size(0)] = text
            text_lengths[i] = text.size(0)

            cln_spec = row[1]
            cln_spec_padded[i, :, :cln_spec.size(1)] = cln_spec
            spec_lengths[i] = cln_spec.size(1)

            cln_wav = row[2]
            cln_wav_padded[i, :, :cln_wav.size(1)] = cln_wav
            wav_lengths[i] = cln_wav.size(1)

            cln_embedd[i, :] = row[3]

            env_spec = row[4]
            env_spec_padded[i, :, :env_spec.size(1)] = env_spec    

            env_wav = row[5]
            env_wav_padded[i, :, :env_wav.size(1)] = env_wav

            env_embedd[i, :] = row[6]

        if self.return_ids:
            return text_padded, text_lengths, cln_spec_padded, env_spec_padded, spec_lengths, cln_wav_padded, env_wav_padded, wav_lengths, cln_embedd, env_embedd, ids_sorted_decreasing
        return text_padded, text_lengths, cln_spec_padded, env_spec_padded, spec_lengths, cln_wav_padded, env_wav_padded, wav_lengths, cln_embedd, env_embedd


class DistributedBucketSampler(torch.utils.data.distributed.DistributedSampler):
    """
    Maintain similar input lengths in a batch.
    Length groups are specified by boundaries.
    Ex) boundaries = [b1, b2, b3] -> any batch is included either {x | b1 < length(x) <=b2} or {x | b2 < length(x) <= b3}.
  
    It removes samples which are not included in the boundaries.
    Ex) boundaries = [b1, b2, b3] -> any x s.t. length(x) <= b1 or length(x) > b3 are discarded.
    """
    def __init__(self, dataset, batch_size, boundaries, num_replicas=None, rank=None, shuffle=True):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, shuffle=shuffle) # num_replicas denotes the number of GPUs, rank denotes the GPU id
        self.lengths = dataset.lengths
        self.batch_size = batch_size
        self.boundaries = boundaries
  
        self.buckets, self.num_samples_per_bucket = self._create_buckets()
        self.total_size = sum(self.num_samples_per_bucket)
        self.num_samples = self.total_size // self.num_replicas
  
    def _create_buckets(self):
        buckets = [[] for _ in range(len(self.boundaries) - 1)]
        for i in range(len(self.lengths)):
            length = self.lengths[i]
            idx_bucket = self._bisect(length)
            if idx_bucket != -1:
                buckets[idx_bucket].append(i)
  
        for i in range(len(buckets) - 1, 0, -1):
            if len(buckets[i]) == 0:
                buckets.pop(i)
                self.boundaries.pop(i+1)
  
        num_samples_per_bucket = []
        for i in range(len(buckets)):
            len_bucket = len(buckets[i])
            total_batch_size = self.num_replicas * self.batch_size
            rem = (total_batch_size - (len_bucket % total_batch_size)) % total_batch_size
            num_samples_per_bucket.append(len_bucket + rem)
        return buckets, num_samples_per_bucket
  
    def __iter__(self):
        # deterministically shuffle based on epoch
        g = torch.Generator()
        g.manual_seed(self.epoch)

        indices = []
        if self.shuffle:
            for bucket in self.buckets:
                indices.append(torch.randperm(len(bucket), generator=g).tolist())
        else:
            for bucket in self.buckets:
                indices.append(list(range(len(bucket))))

        batches = []
        for i in range(len(self.buckets)):
            bucket = self.buckets[i]
            len_bucket = len(bucket)
            ids_bucket = indices[i]
            num_samples_bucket = self.num_samples_per_bucket[i]
    
            # add extra samples to make it evenly divisible
            rem = num_samples_bucket - len_bucket
            ids_bucket = ids_bucket + ids_bucket * (rem // len_bucket) + ids_bucket[:(rem % len_bucket)]
    
            # subsample
            ids_bucket = ids_bucket[self.rank::self.num_replicas]
    
            # batching
            for j in range(len(ids_bucket) // self.batch_size):
                batch = [bucket[idx] for idx in ids_bucket[j*self.batch_size:(j+1)*self.batch_size]]
                batches.append(batch)

        if self.shuffle:
            batch_ids = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in batch_ids]
        self.batches = batches
    
        assert len(self.batches) * self.batch_size == self.num_samples
        return iter(self.batches)
  
    def _bisect(self, x, lo=0, hi=None):
        if hi is None:
            hi = len(self.boundaries) - 1
    
        if hi > lo:
            mid = (hi + lo) // 2
            if self.boundaries[mid] < x and x <= self.boundaries[mid+1]:
                return mid
            elif x <= self.boundaries[mid]:
                return self._bisect(x, lo, mid)
            else:
                return self._bisect(x, mid + 1, hi)
        else:
            return -1

    def __len__(self):
        return self.num_samples // self.batch_size

import sys
sys.path.append("..")
import os
import time
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.cuda.amp import autocast, GradScaler

import models.commons as commons
import utils
from datasets.dataset import (
    TextAudioSpeakerLoader,
    TextAudioSpeakerCollate,
    DistributedBucketSampler
)
from models.model import (
    SynthesizerTrn,
    MultiPeriodDiscriminator,
)
from models.losses import (
    generator_loss,
    discriminator_loss,
    feature_loss,
    kl_loss,
    cal_lsd
)
from datasets.mel_processing import mel_spectrogram_torch, spec_to_mel_torch, spectral_normalize_torch
from text.symbols import symbols


torch.backends.cudnn.benchmark = True
global_step = 0


def main():
    """Assume Single Node Multi GPUs Training Only"""
    assert torch.cuda.is_available(), "CPU training is not allowed."

    n_gpus = torch.cuda.device_count()
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '54322'

    hps = utils.get_hparams()
    mp.spawn(run, nprocs=n_gpus, args=(n_gpus, hps,))


def run(rank, n_gpus, hps):
    global global_step
    if rank == 0:
        logger = utils.get_logger(hps.model_dir)
        logger.info(hps)
        utils.check_git_hash(hps.model_dir)
        writer = SummaryWriter(log_dir=os.path.join(hps.model_dir, "train"))
        writer_eval = SummaryWriter(log_dir=os.path.join(hps.model_dir, "eval"))

    dist.init_process_group(backend='nccl', init_method='env://', world_size=n_gpus, rank=rank)
    torch.manual_seed(hps.train.seed)
    torch.cuda.set_device(rank)

    train_dataset = TextAudioSpeakerLoader(hps.data.training_files, hps.data, fix_env=False)

    # Divide the train files with similar length into a bucket.
    train_sampler = DistributedBucketSampler(
        train_dataset,
        hps.train.batch_size,
        [32, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
        num_replicas=n_gpus,
        rank=rank,
        shuffle=True)
    
    # Padding the text, specs, and audios in each mini-batch to the same length.
    collate_fn = TextAudioSpeakerCollate()

    train_loader = DataLoader(train_dataset, num_workers=8, shuffle=False, pin_memory=True,
        collate_fn=collate_fn, batch_sampler=train_sampler)
    if rank == 0:
        eval_dataset = TextAudioSpeakerLoader(hps.data.validation_files, hps.data, fix_env=True)
        eval_loader = DataLoader(eval_dataset, num_workers=8, shuffle=False, batch_size=1, pin_memory=True,
            drop_last=False, collate_fn=collate_fn)

    net_g = SynthesizerTrn(
        len(symbols),
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model).cuda(rank)
    net_d = MultiPeriodDiscriminator(hps.model.use_spectral_norm).cuda(rank)

    optim_g = torch.optim.AdamW(net_g.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    optim_d = torch.optim.AdamW(net_d.parameters(), hps.train.learning_rate, betas=hps.train.betas, eps=hps.train.eps)
    net_g = DDP(net_g, device_ids=[rank])
    net_d = DDP(net_d, device_ids=[rank])

    try:
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "G_*.pth"), net_g, optim_g)
        _, _, _, epoch_str = utils.load_checkpoint(utils.latest_checkpoint_path(hps.model_dir, "D_*.pth"), net_d, optim_d)
        global_step = (epoch_str - 1) * len(train_loader)
    except:
        epoch_str = 1
        global_step = 0

    scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optim_g, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)
    scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optim_d, gamma=hps.train.lr_decay, last_epoch=epoch_str-2)

    scaler = GradScaler(enabled=hps.train.fp16_run)

    for epoch in range(epoch_str, hps.train.epochs + 1):
        if rank==0:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, eval_loader], logger, [writer, writer_eval])
        else:
            train_and_evaluate(rank, epoch, hps, [net_g, net_d], [optim_g, optim_d], [scheduler_g, scheduler_d], scaler, [train_loader, None], None, None)
        scheduler_g.step()
        scheduler_d.step()


def train_and_evaluate(rank, epoch, hps, nets, optims, schedulers, scaler, loaders, logger, writers):
    net_g, net_d = nets
    optim_g, optim_d = optims
    scheduler_g, scheduler_d = schedulers
    train_loader, eval_loader = loaders
    if writers is not None:
        writer, writer_eval = writers

    train_loader.batch_sampler.set_epoch(epoch)
    global global_step

    net_g.train()
    net_d.train()
    for batch_idx, (x, x_lengths, spec_cln, spec_env, spec_lengths, y_cln, y_env, y_lengths, embedd_cln, embedd_env) in enumerate(train_loader):
        if rank == 0:
            start_b = time.time()
        x, x_lengths = x.cuda(rank, non_blocking=True), x_lengths.cuda(rank, non_blocking=True)
        spec_cln, spec_env, spec_lengths = spec_cln.cuda(rank, non_blocking=True), spec_env.cuda(rank, non_blocking=True), spec_lengths.cuda(rank, non_blocking=True)
        y_cln, y_env, y_lengths = y_cln.cuda(rank, non_blocking=True), y_env.cuda(rank, non_blocking=True), y_lengths.cuda(rank, non_blocking=True)
        embedd_cln, embedd_env = embedd_cln.cuda(rank, non_blocking=True), embedd_env.cuda(rank, non_blocking=True)

        with autocast(enabled=hps.train.fp16_run):
            
            y_hat_cln, y_hat_env, y_spec_enhanced, l_length, attn, ids_slice, x_mask, z_mask,\
                (z, z_p, m_p, logs_p, m_q, logs_q) = net_g(x, x_lengths, spec_env, spec_lengths, embedd_env)
            
            mel_cln = spec_to_mel_torch(
                spec_cln, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)
            mel_enhanced = spec_to_mel_torch(
                y_spec_enhanced, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)
            mel_env = spec_to_mel_torch(
                spec_env, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)

            y_mel_cln = commons.slice_segments(mel_cln, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_mel_env = commons.slice_segments(mel_env, ids_slice, hps.train.segment_size // hps.data.hop_length)
            y_hat_mel_cln = mel_spectrogram_torch(
                y_hat_cln.squeeze(1), 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )
            y_hat_mel_env = mel_spectrogram_torch(
                y_hat_env.squeeze(1), 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate, 
                hps.data.hop_length, 
                hps.data.win_length, 
                hps.data.mel_fmin, 
                hps.data.mel_fmax
            )

            y_cln = commons.slice_segments(y_cln, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 
            y_env = commons.slice_segments(y_env, ids_slice * hps.data.hop_length, hps.train.segment_size) # slice 

            # Discriminator
            y_d_hat_r_cln, y_d_hat_g_cln, _, _ = net_d(y_cln, y_hat_cln.detach())
            y_d_hat_r_env, y_d_hat_g_env, _, _ = net_d(y_env, y_hat_env.detach())
            with autocast(enabled=False):
                loss_disc_cln, losses_disc_r_cln, losses_disc_g_cln = discriminator_loss(y_d_hat_r_cln, y_d_hat_g_cln)
                loss_disc_env, losses_disc_r_env, losses_disc_g_env = discriminator_loss(y_d_hat_r_env, y_d_hat_g_env)
                loss_disc_all = loss_disc_cln + loss_disc_env

        optim_d.zero_grad()
        scaler.scale(loss_disc_all).backward()
        scaler.unscale_(optim_d)
        grad_norm_d = commons.clip_grad_value_(net_d.parameters(), None)
        scaler.step(optim_d)

        with autocast(enabled=hps.train.fp16_run):
            # Generator
            y_d_hat_r_cln, y_d_hat_g_cln, fmap_r_cln, fmap_g_cln = net_d(y_cln, y_hat_cln)
            y_d_hat_r_env, y_d_hat_g_env, fmap_r_env, fmap_g_env = net_d(y_env, y_hat_env)
            with autocast(enabled=False):
                loss_dur = torch.sum(l_length.float()) * hps.train.c_dur
                loss_mel_cln = F.l1_loss(y_mel_cln, y_hat_mel_cln) * hps.train.c_mel
                loss_mel_env = F.l1_loss(y_mel_env, y_hat_mel_env) * hps.train.c_mel
                loss_mel_enhanced = F.l1_loss(mel_cln, mel_enhanced) * hps.train.c_mel
                loss_spec_enhanced = F.mse_loss(spec_cln, y_spec_enhanced) * hps.train.c_spec
                loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * hps.train.c_kl

                loss_fm_cln = feature_loss(fmap_r_cln, fmap_g_cln)
                loss_fm_env = feature_loss(fmap_r_env, fmap_g_env)
                loss_gen_cln, losses_gen_cln = generator_loss(y_d_hat_g_cln)
                loss_gen_env, losses_gen_env = generator_loss(y_d_hat_g_env)
                loss_gen_all = loss_gen_cln + loss_gen_env + loss_fm_cln + loss_fm_env + loss_mel_cln + loss_mel_env + loss_spec_enhanced + loss_mel_enhanced + loss_dur + loss_kl
        optim_g.zero_grad()
        scaler.scale(loss_gen_all).backward()
        scaler.unscale_(optim_g)
        grad_norm_g = commons.clip_grad_value_(net_g.parameters(), None)
        scaler.step(optim_g)
        scaler.update()

        if rank==0:
            if global_step % hps.train.log_interval == 0:
                lr = optim_g.param_groups[0]['lr']
                logger.info('Train Epoch: {} [{:.0f}%]'.format(
                    epoch,
                    100. * batch_idx / len(train_loader)))
                logger.info('Steps : {:d}, Gen Loss: {:4.3f}, Clean Mel Loss : {:4.3f}, Environmental Mel Loss : {:4.3f}, Enhanced Mel Loss : {:4.3f}, Enhanced Spec Loss : {:4.3f}, Duration Loss : {:4.3f}, KL Loss : {:4.3f}, s/b : {:4.3f}'.
                            format(global_step, loss_gen_all.item(), loss_mel_cln.item(), loss_mel_env.item(), loss_mel_enhanced.item(), loss_spec_enhanced.item(), loss_dur.item(), loss_kl.item(), time.time() - start_b))
                
                scalar_dict = {"loss/g/total": loss_gen_all, "loss/d/total": loss_disc_all, "learning_rate": lr, "grad_norm/d": grad_norm_d, "grad_norm/g": grad_norm_g}
                scalar_dict.update({"loss/g/fm_cln": loss_fm_cln, "loss/g/fm_env": loss_fm_env, "loss/g/mel_cln": loss_mel_cln, "loss/g/mel_env": loss_mel_env, "loss/g/mel_enhanced": loss_mel_enhanced, "loss/g/spec_enhanced": loss_spec_enhanced, "loss/g/dur": loss_dur, "loss/g/kl": loss_kl})

                # scalar_dict.update({"loss/g/{}".format(i): v for i, v in enumerate(losses_gen)})
                # scalar_dict.update({"loss/d_r/{}".format(i): v for i, v in enumerate(losses_disc_r)})
                # scalar_dict.update({"loss/d_g/{}".format(i): v for i, v in enumerate(losses_disc_g)})
                
                image_dict = { 
                    "slice/mel_org_cln": utils.plot_spectrogram_to_numpy(y_mel_cln[0].data.cpu().numpy()),
                    "slice/mel_org_env": utils.plot_spectrogram_to_numpy(y_mel_env[0].data.cpu().numpy()),
                    "slice/mel_gen_cln": utils.plot_spectrogram_to_numpy(y_hat_mel_cln[0].data.cpu().numpy()), 
                    "slice/mel_gen_env": utils.plot_spectrogram_to_numpy(y_hat_mel_env[0].data.cpu().numpy()), 
                    "all/mel_cln": utils.plot_spectrogram_to_numpy(mel_cln[0].data.cpu().numpy()),
                    "all/mel_env": utils.plot_spectrogram_to_numpy(mel_env[0].data.cpu().numpy()),
                    "all/mel_enhanced": utils.plot_spectrogram_to_numpy(mel_enhanced[0].data.cpu().numpy()),
                    "all/attn": utils.plot_alignment_to_numpy(attn[0,0].data.cpu().numpy())
                }
                utils.summarize(
                    writer=writer,
                    global_step=global_step, 
                    images=image_dict,
                    scalars=scalar_dict)

            if global_step % hps.train.eval_interval == 0:
                evaluate(hps, net_g, eval_loader, writer_eval)

            if global_step % hps.train.checkpoint_interval == 0 and global_step != 0:
                utils.save_checkpoint(net_g, optim_g, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "G_{}.pth".format(global_step)))
                utils.save_checkpoint(net_d, optim_d, hps.train.learning_rate, epoch, os.path.join(hps.model_dir, "D_{}.pth".format(global_step)))
        global_step += 1
    
    if rank == 0:
        logger.info('====> Epoch: {}'.format(epoch))

 
def evaluate(hps, generator, eval_loader, writer_eval):
    generator.eval()
    val_spec_enhanced_err_tot = 0
    val_mel_enhanced_err_tot = 0
    val_lsd_score_tot = 0
    with torch.no_grad():
        for batch_idx, (x, x_lengths, spec_cln, spec_env, spec_lengths, y_cln, y_env, y_lengths, embedd_cln, embedd_env) in enumerate(eval_loader):
            x, x_lengths = x.cuda(0), x_lengths.cuda(0)
            spec_cln, spec_env, spec_lengths = spec_cln.cuda(0), spec_env.cuda(0), spec_lengths.cuda(0)
            y_cln, y_env, y_lengths = y_cln.cuda(0), y_env.cuda(0), y_lengths.cuda(0)
            embedd_cln, embedd_env = embedd_cln.cuda(0), embedd_env.cuda(0)
        
            y_hat_cln, y_hat_env, y_spec_enhanced, attn, mask, *_ = generator.module.infer(x, x_lengths, spec_env, spec_lengths, embedd_env, max_len=1000)
            y_hat_lengths = mask.sum([1,2]).long() * hps.data.hop_length

            y_hat_enhanced, *_ = generator.module.se(spec_env, spec_lengths, embedd_env)

            mel_cln = spec_to_mel_torch(
                spec_cln, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)
            mel_env = spec_to_mel_torch(
                spec_env, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)
            mel_enhanced = spec_to_mel_torch(
                y_spec_enhanced, 
                hps.data.filter_length, 
                hps.data.n_mel_channels, 
                hps.data.sampling_rate,
                hps.data.mel_fmin, 
                hps.data.mel_fmax)
            y_hat_mel_cln = mel_spectrogram_torch(
                y_hat_cln.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )
            y_hat_mel_env = mel_spectrogram_torch(
                y_hat_env.squeeze(1).float(),
                hps.data.filter_length,
                hps.data.n_mel_channels,
                hps.data.sampling_rate,
                hps.data.hop_length,
                hps.data.win_length,
                hps.data.mel_fmin,
                hps.data.mel_fmax
            )

            val_spec_enhanced_err_tot += F.mse_loss(spec_cln, y_spec_enhanced).item()
            val_mel_enhanced_err_tot += F.l1_loss(mel_cln, mel_enhanced).item()
            val_lsd_score_tot += cal_lsd(y_hat_enhanced.squeeze(1), y_cln.squeeze(1)).item()

            if batch_idx == 0:
                image_dict = {
                    "gen/mel_cln": utils.plot_spectrogram_to_numpy(y_hat_mel_cln[0].cpu().numpy()),
                    "gen/mel_env": utils.plot_spectrogram_to_numpy(y_hat_mel_env[0].cpu().numpy()),
                    "gen/mel_enhanced": utils.plot_spectrogram_to_numpy(mel_enhanced[0].cpu().numpy())

                }
                audio_dict = {
                    "gen/audio_env": y_hat_env[0,:,:y_hat_lengths[0]],
                    "gen/audio_cln": y_hat_cln[0,:,:y_hat_lengths[0]],
                    "gen/audio_enhanced": y_hat_enhanced[0,:,:y_lengths[0]]
                }
                if global_step == 0:
                    image_dict.update({"gt/mel_cln": utils.plot_spectrogram_to_numpy(mel_cln[0].cpu().numpy()),
                                       "gt/mel_env": utils.plot_spectrogram_to_numpy(mel_env[0].cpu().numpy())})
                    audio_dict.update({"gt/audio_cln": y_cln[0,:,:y_lengths[0]], 
                                       "gt/audio_env": y_env[0,:,:y_lengths[0]]})

        val_spec_enhanced_err = val_spec_enhanced_err_tot / (batch_idx+1)
        val_mel_enhanced_err = val_mel_enhanced_err_tot / (batch_idx+1)
        val_lsd_score = val_lsd_score_tot / (batch_idx+1)

    scalar_dict = {"loss/g/spec_enhanced": val_spec_enhanced_err, "loss/g/mel_enhanced": val_mel_enhanced_err, "loss/g/lsd_score": val_lsd_score}

    utils.summarize(
        writer=writer_eval,
        global_step=global_step, 
        images=image_dict,
        audios=audio_dict,
        audio_sampling_rate=hps.data.sampling_rate,
        scalars=scalar_dict
    )
    generator.train()

                           
if __name__ == "__main__":
    main()

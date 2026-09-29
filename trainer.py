import time
import models
import torch
import util
import toolbox
from utils.wassersteinLoss import *
import numpy as np
from modules import *
from scipy.fftpack import dct, idct
import torch.nn.functional as F
import torchvision.transforms.functional as TF
import torch.fft
import cv2
from typing import Literal
import os
# import torch.nn as nn

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

import os
import numpy as np
import torch
import cv2

import io
import torchvision.io as tvio

def jpeg_compress_bhwc_np(arr_bchw: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    arr_bhwc: numpy uint8, shape [B,H,W,C], C in {1,3,4}
    return:   numpy uint8, shape [B,H,W,C]
    """
    arr_bhwc = np.transpose(arr_bchw, (0, 2, 3, 1))
    if arr_bhwc.ndim != 4:
        raise ValueError(f"Expect BHWC numpy array, got ndim={arr_bhwc.ndim}")
    if arr_bhwc.dtype != np.uint8:
        arr_bhwc = np.clip(arr_bhwc, 0, 255).astype(np.uint8)

    B, H, W, C = arr_bhwc.shape
    if C not in (1, 3, 4):
        raise ValueError(f"Unsupported channels: {C} (only 1/3/4)")

    out_list = []
    for i in range(B):
        img = arr_bhwc[i]  # HWC uint8
        if C == 4:
            rgb = img[:, :, :3]
            a   = img[:, :, 3:4]
            t_chw = torch.from_numpy(rgb).permute(2, 0, 1).contiguous()  # [3,H,W] u8 CPU
            buf = tvio.encode_jpeg(t_chw, quality=quality)
            dec = tvio.decode_jpeg(buf)  # [3,H,W] u8
            dec_hwc = dec.permute(1, 2, 0).contiguous().numpy()  # [H,W,3]
            out = np.concatenate([dec_hwc, a], axis=2)  # [H,W,4]
        else:
            # C == 1 or 3
            t_chw = torch.from_numpy(img).permute(2, 0, 1).contiguous()  # [C,H,W] u8
            buf = tvio.encode_jpeg(t_chw, quality=quality)
            dec = tvio.decode_jpeg(buf)  # [C,H,W] u8
            out = dec.permute(1, 2, 0).contiguous().numpy()  # [H,W,C]
        out_list.append(out)

    return np.transpose(np.stack(out_list, axis=0), (0, 3, 1, 2))

def save_same_40_lowpass_versions(
    images: torch.Tensor,          
    labels: torch.Tensor,          # [N]
    folder_path: str,
    low_pass_filter,               # (imgs, cutoff, normalized, filter_type) -> Tensor
    num_keep: int = 40,
    cutoffs = [i/10 for i in range(10)],   # 0.0, 0.1, ..., 0.9
    normalized: bool = True,
    filter_type: str = "ideal",
    save_origin: bool = True
):
    os.makedirs(folder_path, exist_ok=True)

    
    N = images.shape[0]
    sel = np.arange(min(num_keep, N))
    imgs_sel = images[sel]         # [K, C, H, W]
    labels_sel = labels[sel]       # [K]

    
    if save_origin:
        origin_dir = os.path.join(folder_path, "origin")
        os.makedirs(origin_dir, exist_ok=True)
        for j in range(imgs_sel.shape[0]):
            img = imgs_sel[j].detach().cpu().numpy()  # [C,H,W]
            img_bgr = np.transpose(img, (1, 2, 0))[:, :, ::-1]  # CHW->HWC, RGB->BGR
            if img_bgr.max() <= 1.0:
                img_bgr = img_bgr * 255.0
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
            lab = int(labels_sel[j])
            idx_global = int(sel[j])
            cv2.imwrite(os.path.join(origin_dir, f"idx{idx_global}_label{lab}.png"), img_bgr)

    
    for cutoff in cutoffs:
        out_dir = os.path.join(folder_path, f"low_{cutoff:.1f}")
        os.makedirs(out_dir, exist_ok=True)

        imgs_low = low_pass_filter(
            imgs_sel, cutoff=float(cutoff),
            normalized=normalized, filter_type=filter_type
        )  # [K, C, H, W]

        for j in range(imgs_low.shape[0]):
            img = imgs_low[j].detach().cpu().numpy()    # [C,H,W]
            img_bgr = np.transpose(img, (1, 2, 0))[:, :, ::-1]  # CHW->HWC, RGB->BGR
            if img_bgr.max() <= 1.0:
                img_bgr = img_bgr * 255.0
            img_bgr = np.clip(img_bgr, 0, 255).astype(np.uint8)
            lab = int(labels_sel[j])
            idx_global = int(sel[j])
            cv2.imwrite(os.path.join(out_dir, f"idx{idx_global}_label{lab}.png"), img_bgr)





def low_pass_filter(
    img: torch.Tensor,
    cutoff: float,
    *,
    normalized: bool = True,
    filter_type: Literal["ideal", "gaussian", "butterworth"] = "ideal",
    butter_order: int = 2,
) -> torch.Tensor:
    
    assert img.ndim == 4, "img must be (B, C, H, W)"
    B, C, H, W = img.shape
    device = img.device
    dtype = img.dtype

    # FFT (centered)
    F = torch.fft.fft2(img, dim=(-2, -1))
    F = torch.fft.fftshift(F, dim=(-2, -1))

    # frequency radius grid 
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    cy = (H // 2)
    cx = (W // 2)
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)  # (H, W)

    # cutoff -> radius in pixel units
    nyquist = min(H, W) / 2.0
    radius = (cutoff * nyquist) if normalized else float(cutoff)
    radius = torch.tensor(radius, device=device, dtype=dtype).clamp(min=1e-6)

    
    if filter_type == "ideal":
        mask = (r <= radius).to(dtype)
    elif filter_type == "gaussian":
        # exp(-(r^2)/(2*radius^2))
        mask = torch.exp(-0.5 * (r / radius) ** 2)
    elif filter_type == "butterworth":
        n = max(1, int(butter_order))
        mask = 1.0 / (1.0 + (r / radius) ** (2 * n))
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    # broadcast to (B, C, H, W)
    mask = mask[None, None, :, :]

    # Apply & inverse FFT
    F_filt = F * mask
    F_filt = torch.fft.ifftshift(F_filt, dim=(-2, -1))
    img_lp = torch.fft.ifft2(F_filt, dim=(-2, -1)).real
    return img_lp
    


def high_pass_filter(
    img: torch.Tensor,
    cutoff: float,
    *,
    normalized: bool = True,
    filter_type: Literal["ideal", "gaussian", "butterworth"] = "ideal",
    butter_order: int = 2,
) -> torch.Tensor:
    """
    Apply a circular high-pass filter in the Fourier domain.

    Args:
        img: (B, C, H, W), float tensor
        cutoff:
            - if normalized=True: cutoff in [0,1], relative to Nyquist radius (min(H,W)/2)
            - if normalized=False: cutoff is pixel radius (consistent with your "Frequency Radius")
        filter_type: "ideal" | "gaussian" | "butterworth"
        butter_order: order for butterworth high-pass

    Returns:
        High-pass filtered image, same shape as input
    """
    assert img.ndim == 4, "img must be (B, C, H, W)"
    B, C, H, W = img.shape
    device, dtype = img.device, img.dtype

    # FFT (centered)
    F = torch.fft.fft2(img, dim=(-2, -1))
    F = torch.fft.fftshift(F, dim=(-2, -1))

    # frequency radius grid
    yy, xx = torch.meshgrid(
        torch.arange(H, device=device, dtype=dtype),
        torch.arange(W, device=device, dtype=dtype),
        indexing="ij",
    )
    cy, cx = H // 2, W // 2
    r = torch.sqrt((yy - cy) ** 2 + (xx - cx) ** 2)

    # cutoff -> radius in pixels
    nyquist = min(H, W) / 2.0
    radius = (cutoff * nyquist) if normalized else float(cutoff)
    radius = torch.tensor(radius, device=device, dtype=dtype).clamp(min=1e-6)

    # different types
    if filter_type == "ideal":
        mask = (r > radius).to(dtype)
    elif filter_type == "gaussian":
        # Gaussian high-pass = 1 - exp(-(r^2)/(2*radius^2))
        mask = 1.0 - torch.exp(-0.5 * (r / radius) ** 2)
    elif filter_type == "butterworth":
        n = max(1, int(butter_order))
        mask = 1.0 / (1.0 + (radius / (r + 1e-6)) ** (2 * n))
    else:
        raise ValueError(f"Unknown filter_type: {filter_type}")

    mask = mask[None, None, :, :]

    # Apply & inverse FFT
    F_filt = F * mask
    F_filt = torch.fft.ifftshift(F_filt, dim=(-2, -1))
    img_hp = torch.fft.ifft2(F_filt, dim=(-2, -1)).real
    return img_hp

def get_frequency_transforms():
    return [
        lambda x: low_pass_filter(x, cutoff=6, normalized=False, filter_type="ideal"),
        lambda x: high_pass_filter(x, cutoff=4, normalized=False, filter_type="ideal"),
    ]


class Trainer():
    def __init__(self, criterion, data_loader, logger, config, global_step=0,
                 target='train_dataset'):
        self.criterion = criterion
        self.data_loader = data_loader
        self.logger = logger
        self.config = config
        self.log_frequency = config.log_frequency if config.log_frequency is not None else 100
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()
        self.global_step = global_step
        self.trip=torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.target = target
        print(self.target)

    def _reset_stats(self):
        self.loss_meters = util.AverageMeter()
        self.acc_meters = util.AverageMeter()
        self.acc5_meters = util.AverageMeter()

    def train(self, args, epoch, model, criterion, optimizer, random_noise=None, use_generator=False):
        model.train()
        for idx, (images, labels) in enumerate(self.data_loader[self.target]):
            
            if getattr(args, 'jpeg_defense', False):
                # print('------Using JPEG compression as defense for clean CIFAR10 train------')
                images = images.mul(255).cpu().numpy()
                images = jpeg_compress_bhwc_np(
                    images, quality=args.img_denoise
                )
                images = torch.tensor(images.astype(np.float32) / 255.0)
            images, labels = images.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            if random_noise is not None :
                random_noise = random_noise.detach().to(device)
                for i in range(len(labels)):
                    class_index = labels[i].item()
                    images[i] += random_noise[class_index].clone()
                    images[i] = torch.clamp(images[i], 0, 1)
                    
          
            if args.filter_type == 'low':   
                images = low_pass_filter(images, cutoff=args.cutoff, normalized=True, filter_type="ideal")     
                
                
            elif args.filter_type == 'high':          
                images = high_pass_filter(images, cutoff=args.cutoff, normalized=True, filter_type="ideal")      
                
            
            start = time.time()
            log_payload = self.train_batch(images, labels, model, optimizer,args)
            end = time.time()
            time_used = end - start
            if self.global_step % self.log_frequency == 0:
                display = util.log_display(epoch=epoch,
                                           global_step=self.global_step,
                                           time_elapse=time_used,
                                           **log_payload)
                self.logger.info(display)
            self.global_step += 1
        return self.global_step
    
    def helper(self, f1, f2, c1, c2, shift):
        with torch.no_grad():
            # Comes straight from backbone which is currently frozen. this saves mem.
            fd = tensor_correlation(norm(f1), norm(f2))

            if self.cfg.pointwise:
                old_mean = fd.mean()
                fd -= fd.mean([3, 4], keepdim=True)
                fd = fd - fd.mean() + old_mean

        cd = tensor_correlation(norm(c1), norm(c2))
        min_val = 0.0
        # if self.cfg.zero_clamp:
        #     min_val = 0.0
        # else:
        #     min_val = -9999.0
        loss = - cd.clamp(min_val) * (fd - shift)
        # if self.cfg.stabalize:
        #     loss = - cd.clamp(min_val, .8) * (fd - shift)
        # else:
        #     loss = - cd.clamp(min_val) * (fd - shift)

        return loss, cd

    def train_batch(self, images, labels, model, optimizer,args):
        model.zero_grad()
        optimizer.zero_grad()
        if isinstance(self.criterion, torch.nn.CrossEntropyLoss) or isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            
            
            _, logits = model(images)
            
            if 'Poison' in args.train_data_type:
                loss =  self.criterion(logits, labels)
            else:    
                loss =  self.criterion(logits, labels)                     
                # loss = loss_f
        else:
            logits, loss = self.criterion(model, images, labels, optimizer)
        if isinstance(self.criterion, models.CutMixCrossEntropyLoss):
            _, labels = torch.max(labels.data, 1)
        loss.backward()  
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), self.config.grad_clip)
        optimizer.step()  
        if logits.shape[1] >= 5:
            acc, acc5 = util.accuracy(logits, labels, topk=(1, 5))
            acc, acc5 = acc.item(), acc5.item()
        else:
            acc, = util.accuracy(logits, labels, topk=(1,))
            acc, acc5 = acc.item(), 1
        self.loss_meters.update(loss.item(), labels.shape[0])
        self.acc_meters.update(acc, labels.shape[0])
        self.acc5_meters.update(acc5, labels.shape[0])
        payload = {"acc": acc,
                   "acc_avg": self.acc_meters.avg,
                   "loss": loss,
                   "loss_avg": self.loss_meters.avg,
                   "lr": optimizer.param_groups[0]['lr'],
                   "|gn|": grad_norm}
        return payload

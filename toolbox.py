import collections
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import torchvision
from models.GAN import Generator, Discriminator
import clip
from PIL import Image
from utils.contrastive_loss import infoNCELoss
from torchvision import transforms
from utils.wassersteinLoss import *
import ipdb
import torch.optim as optim
import pywt

if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')

def l2norm(X, dim=-1, eps=1e-8):
    """L2-normalize columns of X"""
    norm = torch.pow(X, 2).sum(dim=dim, keepdim=True).sqrt() + eps
    X = torch.div(X, norm)
    return X


def spectral_entropy_regularization(perturbation, lambda_entropy=1.0, eps=1e-10):
    
    fft_perturbation = torch.fft.fft2(perturbation)
    
    
    fft_shifted = torch.fft.fftshift(fft_perturbation)
    
    
    power_spectrum = torch.abs(fft_shifted) ** 2
    
    
    normalized_power = power_spectrum / (torch.sum(power_spectrum) + eps)
    
    
    entropy = -torch.sum(normalized_power * torch.log(normalized_power + eps))
    
    
    spec_loss = lambda_entropy * entropy
    
    return spec_loss

def kl_divergence(p_logits, q_logits):
    """
    Computes D_KL(p || q), where:
    - p_logits: logits from F_i(x)
    - q_logits: logits from F_i(x + δ)
    """

    # Convert logits to probabilities
    p = F.softmax(p_logits, dim=1)              # p_theta(F_i(x))
    q = F.softmax(q_logits, dim=1)              # p_theta(F_i(x + δ))

    # To avoid log(0), add small epsilon
    eps = 1e-8
    p = p.clamp(min=eps)
    q = q.clamp(min=eps)

    # Compute KL divergence for each sample
    kl = torch.sum(p * torch.log(p / q), dim=1)

    # Return mean over batch
    return kl.mean()

def _cosine01(u: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Cosine similarity in [0,1], per-sample scalar."""
    cos = F.cosine_similarity(u, v, dim=1).clamp(-1, 1)
    return (cos + 1.0) / 2.0

def _ssim_scalar(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """
    Return SSIM per sample (B,). Prefer kornia; fallback to pixel-cosine if unavailable.
    a,b: (B,C,H,W) in [0,1]
    """
    try:
        import kornia.metrics as km
        
        ssim_map = km.ssim(a, b, window_size=11, reduction='none')  # (B,C,H,W)
        return ssim_map.mean(dim=(1, 2, 3))  # (B,)
    except Exception:
        
        af = a.view(a.size(0), -1)
        bf = b.view(b.size(0), -1)
        return _cosine01(af, bf)

def _hybrid_sim(x: torch.Tensor, x_p: torch.Tensor, model) -> torch.Tensor:
    """
    sim(a,b) = cos(phi(a), phi(b)) + SSIM(a,b)  -> (B,)
    """
    # feature-level
    feat_x,_  = model(x)    # (B,D)
    feat_xp,_ = model(x_p)  # (B,D)
    cos_feat = _cosine01(feat_x, feat_xp)  # (B,)

    # perceptual (image-domain)
    ssim_val = _ssim_scalar(x, x_p)        # (B,)

    return cos_feat  + ssim_val           # (B, 

def cosine_consistency(p, q):
    """
    Computes 1 - cosine similarity between features f1 and f2.
    Inputs: f1, f2
    """
    p = F.normalize(p, dim=1)
    q = F.normalize(q, dim=1)
    loss = 1.0 - torch.sum(p * q, dim=1)  # shape [B]
    return loss.mean()

import torch
import random

# def entropy

def continuous_band_mask(img_tensor, mask_ratio=0.2, mode='random_band'):
    """
    Apply a continuous band mask in the frequency domain.
    mask_ratio: e.g., 0.2 means 20% of frequencies will be masked (as a contiguous band)
    mode: 'low' | 'mid' | 'high' | 'random_band'
    """
    B, C, H, W = img_tensor.shape
    fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    fft = torch.fft.fftshift(fft)

    # Build frequency radius grid
    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    freq_radius = ((yy - center_y) ** 2 + (xx - center_x) ** 2).float().sqrt().to(img_tensor.device)

    # Flatten and sort frequency indices by radius
    flat_radius = freq_radius.flatten()
    sorted_indices = torch.argsort(flat_radius)
    total_freqs = H * W
    num_mask = int(total_freqs * mask_ratio)

    if mode == 'low':
        masked_indices = sorted_indices[:num_mask]
    elif mode == 'high':
        masked_indices = sorted_indices[-num_mask:]
    elif mode == 'mid':
        mid_start = total_freqs // 2 - num_mask // 2
        masked_indices = sorted_indices[mid_start:mid_start + num_mask]
    elif mode == 'random_band':
        start = random.randint(0, total_freqs - num_mask)
        masked_indices = sorted_indices[start:start + num_mask]
    else:
        raise ValueError("Invalid mode. Choose from ['low', 'mid', 'high', 'random_band'].")

    # Create binary mask: 1 = keep, 0 = drop
    mask = torch.ones(H * W, device=img_tensor.device)
    mask[masked_indices] = 0
    mask = mask.reshape(1, 1, H, W)

    # Apply mask
    fft_masked = fft * mask
    fft_masked = torch.fft.ifftshift(fft_masked)
    img_filtered = torch.fft.ifft2(fft_masked, dim=(-2, -1)).real

    return img_filtered


def split_frequency(img_tensor, cutoff_ratio=0.3):
    """
    Split an image tensor into low-frequency and high-frequency components.
    """
    B, C, H, W = img_tensor.shape
    fft = torch.fft.fft2(img_tensor, dim=(-2, -1))
    fft_shifted = torch.fft.fftshift(fft)

    yy, xx = torch.meshgrid(torch.arange(H), torch.arange(W), indexing='ij')
    center_y, center_x = H // 2, W // 2
    radius = ((yy - center_y) ** 2 + (xx - center_x) ** 2).float().sqrt().to(img_tensor.device)
    max_radius = radius.max()
    mask_low = (radius <= max_radius * cutoff_ratio).float()
    mask_high = 1 - mask_low

    mask_low = mask_low[None, None, :, :]
    mask_high = mask_high[None, None, :, :]

    fft_low = fft_shifted * mask_low
    fft_high = fft_shifted * mask_high

    low_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_low), dim=(-2, -1)).real
    high_freq = torch.fft.ifft2(torch.fft.ifftshift(fft_high), dim=(-2, -1)).real

    return low_freq, high_freq

def frequency_guided_unlearnable_loss(args, clean_img, perturbed_img, model, target_label, alpha=1.0, beta=0.5):
    """
    clean_img: [B, C, H, W] original image
    perturbed_img: [B, C, H, W] image with perturbation
    model: classification model
    target_label: labels (used for classification loss)
    alpha, beta: weight parameters
    """
    # Split into frequency components
    clean_low, clean_high = split_frequency(clean_img, cutoff_ratio=args.cutoff)
    perturbed_low, perturbed_high = split_frequency(perturbed_img, cutoff_ratio=args.cutoff)

    # Classification loss on high-frequency part (encourage unlearnability)
    f_high, high_logits = model(perturbed_high)
    f_low, low_logits = model(perturbed_low)
    # Apply softmax
    flat_high = F.normalize(high_logits, dim=1)
    flat_low = F.normalize(low_logits.detach(), dim=1)
    # loss_high = F.cosine_similarity(flat_high, flat_low, dim=1).mean()
    pseudo_target = low_logits.detach().argmax(dim=1)  
    loss_high = F.cross_entropy(high_logits, pseudo_target)
    # loss_high = torch.max(loss_high, torch.tensor(0.0, device=clean_img.device))  # Ensure non-negative

    s_low  = _hybrid_sim(clean_low,  perturbed_low, model)   # (B,)
    s_high = _hybrid_sim(clean_high, perturbed_high, model)  # (B,)

    loss_struct = F.mse_loss(s_high, s_low.detach())

    # Total loss
    loss = alpha * loss_high + beta * loss_struct

    
    return loss, {
        'loss_high': loss_high.item(),
        'loss_struct': loss_struct.item()
    }

def get_frequency_transforms():
    return [
        # lambda x: continuous_band_mask(x, mask_ratio=0.2, mode='low'),          
        # lambda x: continuous_band_mask(x, mask_ratio=0.2, mode='high'),         
        # lambda x: continuous_band_mask(x, mask_ratio=0.2, mode='mid'), 
        lambda x: continuous_band_mask(x, mask_ratio=0.2, mode='random_band'),
    ]



def center_crop(batch, crop_size):
    _, _, h, w = batch.size()
    ch, cw = crop_size
    start_h = (h - ch) // 2
    start_w = (w - cw) // 2
    return batch[:, :, start_h:start_h + ch, start_w:start_w + cw]

def preprocess(input):
    resize = transforms.Resize((224, 224))
    normalize = transforms.Normalize(mean=[0.48145466, 0.4578275, 0.40821073], std=[0.26862954, 0.26130258, 0.27577711])

    resized_batch = F.interpolate(input, size=(224, 224), mode='bicubic', align_corners=False)
    cropped_batch = center_crop(resized_batch, (224, 224))
    normalized_batch = normalize(cropped_batch)
    return normalized_batch

class PerturbationTool():
    def __init__(self, args, seed=0, epsilon=0.03137254901, num_steps=20, step_size=0.00784313725):
        self.epsilon = epsilon
        self.num_steps = num_steps
        self.step_size = step_size
        self.seed = seed
        self.generator = Generator(args).cuda()
        
        # self.filter = TrainableHighPassFilter().cuda()
        
        self.optimizer_G = torch.optim.Adam(self.generator.parameters(), lr=0.001)
        
        # self.optimizer_F = torch.optim.Adam(self.filter.parameters(), lr=0.001)
        np.random.seed(seed)
        # self.compute_loss = infoNCELoss(tau=5)
        self.trip=torch.nn.TripletMarginLoss(margin=1.0, p=2, eps=1e-7)
        self.frequency_transforms = get_frequency_transforms()
        

    def random_noise(self, noise_shape=[10, 3, 32, 32]):
        random_noise = torch.FloatTensor(*noise_shape).uniform_(-self.epsilon, self.epsilon).to(device)
        return random_noise

    
    def min_min_attack(self, args, images, labels, label_emb, model, optimizer, criterion, random_noise=None, sample_wise=False):
        image = images
        lable = labels
        for _ in range(self.num_steps):
            
            
            noise = self.generator(image)   
            ori_img = image
            
            temp_class_noise = collections.defaultdict(list)
            currentbatch_class_noise = torch.zeros(*args.noise_shape)
            for i in range(len(noise)):
                temp_class_noise[labels[i].item()].append(noise[i])
            for key in temp_class_noise:
                currentbatch_class_noise[key] = torch.stack(temp_class_noise[key]).mean(dim=0)

            noise = torch.stack([currentbatch_class_noise[label.item()] for label in labels]).cuda()

            noise = noise * self.epsilon

            perturb_img = torch.clamp(image + noise, 0, 1)

            # classification loss
            loss_fre_classification = 0
            
            for F_i in self.frequency_transforms:
                x_filtered = F_i(perturb_img)
                f_per, logits = model(x_filtered)
                
                loss_fre_classification = criterion(logits, labels).requires_grad_() +loss_fre_classification
                
            loss_fre_classification = loss_fre_classification / len(self.frequency_transforms)  
            
            
            
            loss_spec = spectral_entropy_regularization(noise, lambda_entropy=0.1)  

            
            loss_fre, log_dict = frequency_guided_unlearnable_loss(
                clean_img=ori_img,
                perturbed_img=perturb_img,
                model=model,
                target_label=labels,
                alpha=1.0,         
                beta=args.beta,           
                args=args  
            )


            
            loss_G =   loss_fre_classification  +loss_fre + loss_spec  
            
            self.optimizer_G.zero_grad()
            loss_G.backward()
            self.optimizer_G.step()

            

        return perturb_img, noise, loss_fre, loss_fre_classification, loss_spec
        

    def min_max_attack(self, images, labels, model, optimizer, criterion, random_noise=None, sample_wise=False):
        if random_noise is None:
            random_noise = torch.FloatTensor(*images.shape).uniform_(-self.epsilon, self.epsilon).to(device)

        perturb_img = Variable(images.data + random_noise, requires_grad=True)
        perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)
        eta = random_noise
        for _ in range(self.num_steps):
            opt = torch.optim.SGD([perturb_img], lr=1e-3)
            opt.zero_grad()
            model.zero_grad()
            if isinstance(criterion, torch.nn.CrossEntropyLoss):
                logits = model(perturb_img)
                loss = criterion(logits, labels)
            else:
                logits, loss = criterion(model, perturb_img, labels, optimizer)
            loss.backward()

            eta = self.step_size * perturb_img.grad.data.sign()
            perturb_img = Variable(perturb_img.data + eta, requires_grad=True)
            eta = torch.clamp(perturb_img.data - images.data, -self.epsilon, self.epsilon)
            perturb_img = Variable(images.data + eta, requires_grad=True)
            perturb_img = Variable(torch.clamp(perturb_img, 0, 1), requires_grad=True)

        return perturb_img, eta

    def _patch_noise_extend_to_img(self, noise, image_size=[3, 32, 32], patch_location='center'):
        c, h, w = image_size[0], image_size[1], image_size[2]
        mask = np.zeros((c, h, w), np.float32)
        x_len, y_len = noise.shape[1], noise.shape[1]

        if patch_location == 'center' or (h == w == x_len == y_len):
            x = h // 2
            y = w // 2
        elif patch_location == 'random':
            x = np.random.randint(x_len // 2, w - x_len // 2)
            y = np.random.randint(y_len // 2, h - y_len // 2)
        else:
            raise('Invalid patch location')

        x1 = np.clip(x - x_len // 2, 0, h)
        x2 = np.clip(x + x_len // 2, 0, h)
        y1 = np.clip(y - y_len // 2, 0, w)
        y2 = np.clip(y + y_len // 2, 0, w)
        if type(noise) is np.ndarray:
            pass
        else:
            mask[:, x1: x2, y1: y2] = noise.cpu().numpy()
        return ((x1, x2, y1, y2), torch.from_numpy(mask).to(device))

import copy
import os
import collections
import numpy as np
import torch
import util
import random
import mlconfig
import pandas
from util import onehot, rand_bbox
from torch.utils.data.dataset import Dataset
from functools import partial
from PIL import Image, ImageFilter
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from fast_autoaugment.FastAutoAugment.archive import fa_reduced_cifar10
from fast_autoaugment.FastAutoAugment.augmentations import apply_augment
if torch.cuda.is_available():
    device = torch.device('cuda')
else:
    device = torch.device('cpu')
import cv2
import toolbox
from tqdm import tqdm
import matplotlib.pyplot as plt
from models.unet import *

# import torch
# import numpy as np
# import matplotlib.pyplot as plt
# from torchvision import transforms
# from PIL import Image
import io
import torchvision.io as tvio
# import torch

def jpeg_compress_bhwc_np(arr_bhwc: np.ndarray, quality: int = 90) -> np.ndarray:
    """
    arr_bhwc: numpy uint8, shape [B,H,W,C], C in {1,3,4}
    return:   numpy uint8, shape [B,H,W,C]
    """
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

    return np.stack(out_list, axis=0)

def compute_radial_spectrum(delta_np):
    """
    Compute 1D radial frequency spectrum for one image.
    Input: delta_np shape [C, H, W]
    Output: 1D numpy array of radial frequency magnitudes (averaged over channels)
    """
    C, H, W = delta_np.shape
    radial_accum = []

    for c in range(C):
        d = delta_np[c]
        fft = np.fft.fft2(d)
        fft_shifted = np.fft.fftshift(fft)
        magnitude = np.abs(fft_shifted)

        y, x = np.indices((H, W))
        center = (H // 2, W // 2)
        r = np.sqrt((x - center[1])**2 + (y - center[0])**2).astype(np.int32)

        radial_sum = np.bincount(r.ravel(), magnitude.ravel())
        radial_count = np.bincount(r.ravel())
        radial_mean = radial_sum / (radial_count + 1e-8)
        radial_accum.append(radial_mean)

    return np.mean(np.stack(radial_accum, axis=0), axis=0)

def save_fft_spectrum_cv2_batch(image_np, perturbed_np, save_dir="visualization/fft_curve_lsp", max_batch=20):
    """
    image_np, perturbed_np: numpy arrays of shape [B, C, H, W]
    Only the first `max_batch` samples will be averaged to compute the FFT spectrum.
    """
    os.makedirs(save_dir, exist_ok=True)

    B = min(max_batch, image_np.shape[0])
    delta_spec_list = []
    clean_spec_list = []
    pert_spec_list = []

    for i in range(B):
        img = image_np[i]
        pert = perturbed_np[i]
        delta = pert - img

        delta_spec_list.append(compute_radial_spectrum(delta))
        clean_spec_list.append(compute_radial_spectrum(img))
        pert_spec_list.append(compute_radial_spectrum(pert))

    mean_delta = np.mean(np.stack(delta_spec_list, axis=0), axis=0)
    mean_clean = np.mean(np.stack(clean_spec_list, axis=0), axis=0)
    mean_pert = np.mean(np.stack(pert_spec_list, axis=0), axis=0)
    
    
    delta_save_path = os.path.join(save_dir, f"delta_spectrum_avg{B}.npy")
    np.save(delta_save_path, mean_delta)
    print(f"[✓] Saved δ radial spectrum data to {delta_save_path}")

    # --- Figure 1: clean vs perturbed vs diff
    plt.figure(figsize=(6, 4))
    plt.plot(mean_clean, label='Original', linestyle='--')
    plt.plot(mean_pert, label='Perturbed', linestyle='-')
    plt.plot(mean_delta, label='Perturbation Δ', linestyle=':')
    plt.xlabel("Frequency Radius")
    plt.ylabel("Average Magnitude")
    plt.title(f"FFT Spectrum Comparison (Avg over {B} samples)")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fft_compare_avg{B}.png"))
    plt.close()

    # --- Figure 2: only Δ-spectrum
    plt.figure(figsize=(5, 3.5))
    plt.plot(mean_delta)
    plt.xlabel("Frequency Radius")
    plt.ylabel("Perturbation Spectrum Magnitude")
    plt.title(f"Perturbation Spectrum (Avg over {B} samples)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(save_dir, f"fft_delta_avg{B}.png"))
    plt.close()
            
            
# Datasets
transform_options = {
    "CIFAR10": {
        "train_transform": [
            transforms.RandomCrop(32, padding=4),
                            transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(10),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    "CIFAR100": {
         "train_transform": [transforms.RandomCrop(32, padding=4),
                             transforms.RandomHorizontalFlip(),
                             transforms.RandomRotation(20),
                             transforms.ToTensor()],
         "test_transform": [transforms.ToTensor()]},
    "SVHN": {
        "train_transform": [transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
    
    "Flower" : {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},

    "ImageNet": {
        "train_transform": [transforms.RandomResizedCrop(224),
                            transforms.RandomHorizontalFlip(),
                            transforms.ColorJitter(brightness=0.4,
                                                   contrast=0.4,
                                                   saturation=0.4,
                                                   hue=0.2),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize(256),
                           transforms.CenterCrop(224),
                           transforms.ToTensor()]},
    "TinyImageNet": {
        "train_transform": [transforms.CenterCrop(256),
                            transforms.Resize((32, 32)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CatDog': {
        "train_transform": [transforms.Resize((32, 32)),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((32, 32)),
                           transforms.ToTensor()]},
    'CelebA': {
        "train_transform": [transforms.CenterCrop((128, 128)),
                            transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.CenterCrop((128, 128)),
                           transforms.ToTensor()]},
    'FaceScrub': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.Resize((128, 128)),
                           transforms.ToTensor()]},
    'WebFace': {
        "train_transform": [transforms.RandomHorizontalFlip(),
                            transforms.ToTensor()],
        "test_transform": [transforms.ToTensor()]},
}
transform_options['PoisonCIFAR10'] = transform_options['CIFAR10']
transform_options['PoisonCIFAR100'] = transform_options['CIFAR100']
transform_options['PoisonCIFAR101'] = transform_options['CIFAR100']
transform_options['PoisonSVHN'] = transform_options['SVHN']
transform_options['PoisonFlower'] = transform_options['Flower']
transform_options['ImageNetMini'] = transform_options['ImageNet']
transform_options['PoisonImageNetMini'] = transform_options['ImageNet']
transform_options['CelebAMini'] = transform_options['CelebA']

def compute_class_mean_noises(args, root, split, generator_ckpt):
    
    composed_transform = transforms.Compose(transform_options[args.train_data_type]['train_transform'])
    raw_ds = ImageNetMini(root=root, split=split, transform=composed_transform)
    # loader = DataLoader(raw_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    if split=='train':
        batch_size=args.train_batch_size
        data_loader = DataLoader(raw_ds, batch_size=args.train_batch_size, shuffle=True, num_workers=args.num_of_workers)
    
    gen_tool = toolbox.PerturbationTool(args).generator
    gen_tool.load_state_dict(torch.load(generator_ckpt, map_location=device))
    gen_tool.eval()
    print("Generator %s loaded!" % (args.generator_filepath))

    
    num_classes = len(raw_ds.classes)
    sums   = {c: np.zeros((3, 224, 224), dtype=np.float64) for c in range(num_classes)}
    counts = {c: 0 for c in range(num_classes)}

    
    with torch.no_grad():
        for imgs, labels in data_loader:
            imgs = imgs.to(device)                   # [B,3,224,224]
            noises = gen_tool(imgs)                  # [B,3,224,224]
            noises = noises.mul(255).clamp_(-255, 255).detach().cpu().numpy()   

            for b in range(noises.shape[0]):
                c = int(labels[b])
                sums[c]   += noises[b]
                counts[c] += 1

    
    class_noise = np.stack([ (sums[c] / counts[c]).astype(np.float32) for c in range(num_classes) ], axis=0)
    class_noise = np.transpose(class_noise, (0, 2, 3, 1))  # [100, 224, 224, 3]
    return class_noise

@mlconfig.register
class DatasetGenerator():
    def __init__(self, args, train_batch_size=128, eval_batch_size=256, num_of_workers=4,
                 train_data_path='../datasets/', train_data_type='CIFAR10', seed=0,
                 test_data_path='../datasets/', test_data_type='CIFAR10', fa=False,
                 no_train_augments=False, poison_rate=1.0, perturb_type='classwise',
                 perturb_tensor_filepath=None, patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None,
                 use_cutout=None, use_cutmix=False, use_mixup=False, use_generator=False, class_noise=None):

        np.random.seed(seed)
        self.train_batch_size = train_batch_size
        self.eval_batch_size = eval_batch_size
        self.num_of_workers = num_of_workers
        self.seed = seed
        self.train_data_type = train_data_type
        self.test_data_type = test_data_type
        self.train_data_path = train_data_path
        self.test_data_path = test_data_path
        self.use_generator = use_generator
        self.class_noise = class_noise

        train_transform = transform_options[train_data_type]['train_transform']
        test_transform = transform_options[test_data_type]['test_transform']
        train_transform = transforms.Compose(train_transform)
        test_transform = transforms.Compose(test_transform)
        if no_train_augments:  # default: False
            train_transform = test_transform

        if fa:  # default: False
            # FastAutoAugment
            train_transform.transforms.insert(0, Augmentation(fa_reduced_cifar10()))
        elif use_cutout is not None: # default: False
            print('Using Cutout')
            train_transform.transforms.append(Cutout(16))

# Training Datasets
        if train_data_type == 'CIFAR10':
            num_of_classes = 10
            train_dataset = datasets.CIFAR10(root=train_data_path, train=True,
                                             download=True, transform=train_transform)
            if getattr(args, 'jpeg_defense', False):
                print('------Using JPEG compression {} as defense for clean CIFAR10 train------'.format(args.img_denoise))
            #     train_dataset.data = jpeg_compress_bhwc_np(
            #         train_dataset.data, quality=args.img_denoise
            #     )
            # test_dataset = datasets.CIFAR10(root=train_data_path, train=True,
            #                                  download=True, transform=test_transform)
            # train_dataset = CustomCIFAR10(root=train_data_path, train=True,
            #                   download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR10':
            num_of_classes = 10
            train_dataset = PoisonCIFAR10(args=args, root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx, use_generator=use_generator)
            test_dataset = PoisonCIFAR10(args=args, root=train_data_path, transform=train_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise,
                                          poison_classwise_idx=poison_classwise_idx, use_generator=use_generator)
            if getattr(args, 'jpeg_defense', False):
                print('------Using JPEG compression {} as defense for UE CIFAR10 train------'.format(args.img_denoise))
        elif train_data_type == 'CIFAR100':
            num_of_classes = 100
            train_dataset = datasets.CIFAR100(root=train_data_path, train=True,
                                              download=True, transform=train_transform)
            # train_dataset = CustomCIFAR100(root=train_data_path, train=True,
            #                   download=True, transform=train_transform)
        elif train_data_type == 'PoisonCIFAR100':
            num_of_classes = 100
            train_dataset = PoisonCIFAR100(args=args, root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise, use_generator=use_generator)
        elif train_data_type == 'PoisonCIFAR101':
            num_of_classes = 101
            poison_cifar10 = PoisonCIFAR10(root=train_data_path, transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise,
                                           poison_classwise_idx=poison_classwise_idx)
            train_dataset = PoisonCIFAR101(train_data_path, split='poison_train',
                                           transform=train_transform, seed=0,
                                           poisn_cifar10_data=poison_cifar10)
        elif train_data_type == 'SVHN':
            num_of_classes = 10
            train_dataset = datasets.SVHN(root=train_data_path, split='train',
                                          download=True, transform=train_transform)
            # train_dataset = CustomSVHN(root=train_data_path, split='train',
            #                               download=True, transform=train_transform)
        elif train_data_type == 'PoisonSVHN':
            num_of_classes = 10
            train_dataset = PoisonSVHN(args=args, root=train_data_path, split='train', download=True, transform=train_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise, use_generator=use_generator)

        elif train_data_type == 'TinyImageNet':
            num_of_classes = 1000
            train_dataset = datasets.ImageNet(root=train_data_path, split='train',
                                              transform=train_transform)
        elif train_data_type == 'ImageNetMini':
            num_of_classes = 100
            train_data_path_imagenetmini = os.path.join(train_data_path,"ImageNet")
            
            train_dataset = ImageNetMini(root=train_data_path_imagenetmini, split='train',
                                         transform=train_transform)
            if getattr(args, 'jpeg_defense', False):
                print('------Using JPEG compression {} as defense for clean ImageNet train------'.format(args.img_denoise))
        elif train_data_type == 'PoisonImageNetMini':
            num_of_classes = 100
            train_data_path_imagenetmini = os.path.join(train_data_path,"ImageNet")
            train_dataset = PoisonImageNetMini(args=args, root=train_data_path_imagenetmini, split='train', transform=train_transform,
                                           poison_rate=poison_rate, perturb_type=perturb_type,
                                           patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                           perturb_tensor_filepath=perturb_tensor_filepath,
                                           add_uniform_noise=add_uniform_noise,
                                           poison_classwise=poison_classwise, use_generator=self.use_generator, class_mean_noise=class_noise)
            if getattr(args, 'jpeg_defense', False):
                print('------Using JPEG compression {} as defense for UE ImageNet train------'.format(args.img_denoise))
        elif train_data_type == 'CatDog':
            train_dataset = CatDogDataset(root=train_data_path, split='train',
                                          transform=train_transform)
        elif train_data_type == 'CelebAMini':
            train_dataset = CelebAMini(root=train_data_path, split="all",
                                       target_type="identity", transform=train_transform)
            test_dataset = CelebAMini(root=train_data_path, split="all",
                                      target_type="identity", transform=test_transform)
        elif train_data_type == 'WebFace':
            train_dataset = datasets.ImageFolder(root=train_data_path, transform=train_transform)
            test_dataset = datasets.ImageFolder(root=test_data_path, transform=test_transform)
        elif train_data_type == 'CelebA':
            train_dataset = datasets.CelebA(root=train_data_path, split="all",
                                            target_type="identity", transform=train_transform)
            test_dataset = datasets.CelebA(root=train_data_path, split="all",
                                           target_type="identity", transform=test_transform)
        else:
            raise('Training Dataset type %s not implemented' % train_data_type)

        # Test Datset
        if test_data_type == 'CIFAR10':
            test_dataset = datasets.CIFAR10(root=test_data_path, train=False,
                                            download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR10':
            test_dataset = PoisonCIFAR10(root=test_data_path, train=False, transform=test_transform,
                                         poison_rate=poison_rate, perturb_type=perturb_type,
                                         patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                         perturb_tensor_filepath=perturb_tensor_filepath,
                                         add_uniform_noise=add_uniform_noise,
                                         poison_classwise=poison_classwise,
                                         poison_classwise_idx=poison_classwise_idx)

        elif test_data_type == 'CIFAR100':
            test_dataset = datasets.CIFAR100(root=test_data_path, train=False,
                                             download=True, transform=test_transform)
        elif test_data_type == 'PoisonCIFAR100':
            test_dataset = PoisonCIFAR100(root=test_data_path, train=False, transform=test_transform,
                                          poison_rate=poison_rate, perturb_type=perturb_type,
                                          patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                          perturb_tensor_filepath=perturb_tensor_filepath,
                                          add_uniform_noise=add_uniform_noise,
                                          poison_classwise=poison_classwise)
        elif test_data_type == 'PoisonCIFAR101':
            test_dataset = PoisonCIFAR101(test_data_path, split='test',
                                          transform=test_transform, seed=0,
                                          poisn_cifar10_data=poison_cifar10)
        elif test_data_type == 'SVHN':
            test_dataset = datasets.SVHN(root=test_data_path, split='test',
                                         download=True, transform=test_transform)
        elif test_data_type == 'PoisonSVHN':
            test_dataset = PoisonSVHN(root=test_data_path, download=True, split='test', transform=test_transform,
                                       poison_rate=poison_rate, perturb_type=perturb_type,
                                       patch_location=patch_location, seed=seed, img_denoise=img_denoise,
                                       perturb_tensor_filepath=perturb_tensor_filepath,
                                       add_uniform_noise=add_uniform_noise,
                                       poison_classwise=poison_classwise)
        elif test_data_type == 'ImageNetMini':
            test_data_path_imagemin = os.path.join(train_data_path,"ImageNet")
            test_dataset = ImageNetMini(root=test_data_path_imagemin, split='val',
                                        transform=test_transform)
        elif test_data_type == 'TinyImageNet':
            test_dataset = datasets.ImageNet(root=test_data_path, split='val',
                                             transform=test_transform)
        elif test_data_type == 'PoisonImageNetMini':
            test_data_path_pimagemin = os.path.join(train_data_path,"ImageNet")
            test_dataset = PoisonImageNetMini(root=test_data_path_pimagemin, split='val', seed=0,
                                              transform=test_transform, poison_rate=poison_rate,
                                              perturb_tensor_filepath=perturb_tensor_filepath)
        elif test_data_type == 'CatDog':
            # Cat Dog only used for transfer exp, no test dataset
            test_dataset = CatDogDataset(root=train_data_path, split='train',
                                         transform=train_transform)
        elif test_data_type == 'CelebAMini' or 'CelebA':
            pass
        elif test_data_type == 'FaceScrub' or test_data_type == 'WebFace':
            pass
        else:
            raise('Test Dataset type %s not implemented' % test_data_type)

        if use_cutmix:  # default: False
            train_dataset = CutMix(dataset=train_dataset, num_class=num_of_classes)
        elif use_mixup:  # default: False
            train_dataset = MixUp(dataset=train_dataset, num_class=num_of_classes)

        self.datasets = {
            'train_dataset': train_dataset,
            'test_dataset': test_dataset,
        }
        return

    def getDataLoader(self, train_shuffle=True, train_drop_last=True):
        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        return data_loaders

    def _split_validation_set(self, train_portion, train_shuffle=True, train_drop_last=True):
        
        np.random.seed(self.seed)
        train_subset = copy.deepcopy(self.datasets['train_dataset'])
        valid_subset = copy.deepcopy(self.datasets['train_dataset'])

        if self.train_data_type == 'ImageNet' or self.train_data_type == 'ImageNetMini' or self.train_data_type == 'TinyImageNet' or self.train_data_type == 'PoisonImageNetMini':
            data, targets = list(zip(*self.datasets['train_dataset'].samples))
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.samples = list(zip(train_D, train_L))
            valid_subset.samples = list(zip(valid_D, valid_L))
        elif self.train_data_type == 'SVHN':
            data, targets = self.datasets['train_dataset'].data, self.datasets['train_dataset'].labels
            datasplit = train_test_split(data, targets, test_size=1-train_portion,
                                         train_size=train_portion, shuffle=True, stratify=targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.labels = train_L
            valid_subset.labels = valid_L
        else:
            datasplit = train_test_split(self.datasets['train_dataset'].data,
                                         self.datasets['train_dataset'].targets,
                                         test_size=1-train_portion, train_size=train_portion,
                                         shuffle=True, stratify=self.datasets['train_dataset'].targets)
            train_D, valid_D, train_L, valid_L = datasplit
            print('Train Labels: ', np.array(train_L))
            print('Valid Labels: ', np.array(valid_L))
            train_subset.data = np.array(train_D)
            valid_subset.data = np.array(valid_D)
            train_subset.targets = train_L
            valid_subset.targets = valid_L

        self.datasets['train_subset'] = train_subset
        self.datasets['valid_subset'] = valid_subset
        print(self.datasets)

        data_loaders = {}

        data_loaders['train_dataset'] = DataLoader(dataset=self.datasets['train_dataset'],
                                                   batch_size=self.train_batch_size,
                                                   shuffle=train_shuffle, pin_memory=True,
                                                   drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['test_dataset'] = DataLoader(dataset=self.datasets['test_dataset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)

        data_loaders['train_subset'] = DataLoader(dataset=self.datasets['train_subset'],
                                                  batch_size=self.train_batch_size,
                                                  shuffle=train_shuffle, pin_memory=True,
                                                  drop_last=train_drop_last, num_workers=self.num_of_workers)

        data_loaders['valid_subset'] = DataLoader(dataset=self.datasets['valid_subset'],
                                                  batch_size=self.eval_batch_size,
                                                  shuffle=False, pin_memory=True,
                                                  drop_last=False, num_workers=self.num_of_workers)
        return data_loaders


def patch_noise_extend_to_img(noise, image_size=[32, 32, 3], patch_location='center'):
    h, w, c = image_size[0], image_size[1], image_size[2]
    mask = np.zeros((h, w, c), np.float32)
    x_len, y_len = noise.shape[0], noise.shape[1]

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
    mask[x1: x2, y1: y2, :] = noise
    return mask

Image_transform = transforms.Compose([transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(10),
            transforms.ToTensor()])
origin_transform = transforms.Compose([transforms.ToTensor()])

class CustomCIFAR10(datasets.CIFAR10):
    def __getitem__(self, index):

        
        img, target = self.data[index], self.targets[index]
        
        img = transforms.ToPILImage()(img)
        
        augmented_img = Image_transform(img)
        
        original_img = Image_transform(img)
        return original_img, augmented_img, target
    
class CustomCIFAR100(datasets.CIFAR100):
    def __getitem__(self, index):

        
        img, target = self.data[index], self.targets[index]
        
        img = transforms.ToPILImage()(img)
        
        augmented_img = Image_transform(img)
        
        original_img = Image_transform(img)
        return original_img, augmented_img, target
    
class PoisonCIFAR10(datasets.CIFAR10):
    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, poison_classwise_idx=None, use_generator=False):
        super(PoisonCIFAR10, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        # whether use generator to produce noise
        if use_generator:
            if args.gue:
                self.generator = UNet(3).cuda()
                self.generator.load_state_dict(torch.load(args.generator_filepath))
            else:
                self.noise_generator = toolbox.PerturbationTool(args,
                                        epsilon=None,
                                        num_steps=None,
                                        step_size=None)
                self.generator = self.noise_generator.generator
                ckpt = torch.load(args.generator_filepath, map_location=device)
                model_state = self.generator.state_dict()
                filtered_ckpt = {k: v for k, v in ckpt.items() 
                    if k in model_state and v.shape == model_state[k].shape}
                model_state.update(filtered_ckpt)
                self.generator.load_state_dict(model_state, strict=False)
            
            print("Generator %s loaded!" % (args.generator_filepath))
            self.generator.eval()

            self.patch_location = patch_location
            self.img_denoise = img_denoise
            self.data = self.data.astype(np.float32)
            ori_data = copy.deepcopy(self.data)  

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise: # default： False
                targets = list(range(0, 10))
                if poison_classwise_idx is None:
                    self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                else:
                    self.poison_class = poison_classwise_idx
                self.poison_samples_idx = []
                for i, label in enumerate(self.targets):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            class_noises = {class_id: [] for class_id in range(len(self.classes))}  
            for idx in self.poison_samples_idx:

                
                

                self.poison_samples[idx] = True
                data_normalize = self.data[idx] / 255.0  
                data_normalize_t = torch.from_numpy(np.transpose(data_normalize, (2, 0, 1))).to(device)
                # img_shape = data_normalize_t.shape
                data_input = torch.unsqueeze(data_normalize_t, 0)
                if perturb_type == 'samplewise':
                    # Sample Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    noise = noise.squeeze(0).mul(255).permute(1, 2, 0).detach().cpu().numpy()  # (32, 32, 3)
                    if args.gue:
                        noise = noise * (8/255)
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    noise = noise.squeeze(0).mul(255).permute(1, 2, 0).detach().cpu().numpy()  # (32, 32, 3)
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    class_noises[self.targets[idx]].append(noise)

            if perturb_type == 'classwise':
                
                mean_class_noises = {}
                for class_id in range(len(self.classes)):
                    if class_noises[class_id]:  
                        mean_class_noises[class_id] = np.mean(class_noises[class_id], axis=0)
                # mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                if args.gue:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                else:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                for idx in self.poison_samples_idx:
                    
                    self.data[idx] = self.data[idx] + mean_class_noises[self.targets[idx]]
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                
            self.data = self.data.astype(np.uint8)
            # import ipdb; ipdb.set_trace()
            
            print('add_uniform_noise: ', add_uniform_noise)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
            
        else:
            # self.perturb_tensor = torch.load(perturb_tensor_filepath, map_location=device)
            self.perturb_tensor = torch.load(args.generator_filepath, map_location=device)
            if not isinstance(self.perturb_tensor, torch.Tensor):
                 self.perturb_tensor = torch.tensor(self.perturb_tensor)
                #  ipdb.set_trace()
                 self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
            
            if self.data.shape[2] != self.perturb_tensor.shape[2]:
                self.perturb_tensor=torch.nn.functional.interpolate(self.perturb_tensor, size=(self.data.shape[2], self.data.shape[2]), mode='bilinear', align_corners=False)
            
            print(self.perturb_tensor)
            if len(self.perturb_tensor.shape) == 4 and self.perturb_tensor.shape[3] != 3:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.patch_location = patch_location
            self.img_denoise = img_denoise
            self.data = self.data.astype(np.float32)
            ori_data = copy.deepcopy(self.data)  

            if perturb_type == 'samplewise': all_len = len(self)
            else: all_len = 10  # classwise
            condition = True
            while condition:
                if len(self.perturb_tensor) < all_len:
                    # tue_len = len(self) - len(self.perturb_tensor)
                    alpha = 0.5
                    rand_noise_1 = np.random.permutation(self.perturb_tensor.shape[0])
                    rand_noise_1 = np.take(self.perturb_tensor, rand_noise_1, axis=0)
                    # ipdb.set_trace()
                    rand_noise_2 = np.random.permutation(rand_noise_1.shape[0])
                    rand_noise_2 = np.take(self.perturb_tensor, rand_noise_2, axis=0)
                    new_noise =alpha*rand_noise_1 + (1-alpha)*rand_noise_2
                    self.perturb_tensor=np.concatenate((self.perturb_tensor, new_noise), axis=0)
                    
                elif len(self.perturb_tensor) > all_len:
                    self.perturb_tensor = self.perturb_tensor[:all_len,:,:,:]
                    condition = False
                elif len(self.perturb_tensor) == all_len:
                    condition = False

            
            
            
            # Check Shape
            target_dim = self.perturb_tensor.shape[0] if len(self.perturb_tensor.shape) == 4 else self.perturb_tensor.shape[1]
            if perturb_type == 'samplewise' and target_dim != len(self):
                raise('Poison Perturb Tensor size not match for samplewise')
            elif perturb_type == 'classwise' and target_dim != 10:
                raise('Poison Perturb Tensor size not match for classwise')

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise:  # default： False
                targets = list(range(0, 10))
                if poison_classwise_idx is None:
                    self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                else:
                    self.poison_class = poison_classwise_idx
                self.poison_samples_idx = []
                for i, label in enumerate(self.targets):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True
                if len(self.perturb_tensor.shape) == 5:
                    perturb_id = random.choice(range(self.perturb_tensor.shape[0]))
                    perturb_tensor = self.perturb_tensor[perturb_id]
                else:
                    perturb_tensor = self.perturb_tensor
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = perturb_tensor[idx]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = perturb_tensor[self.targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                if add_uniform_noise: # default: False
                    noise += np.random.uniform(0, 8, (32, 32, 3))

                
                

                
                self.data[idx] = self.data[idx] + noise
                self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                
                
            self.data = self.data.astype(np.uint8)
            
            print('add_uniform_noise: ', add_uniform_noise)
            print(self.perturb_tensor.shape)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
            


class PoisonCIFAR100(datasets.CIFAR100):
    def __init__(self, args, root, train=True, transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, use_generator=False):
        super(PoisonCIFAR100, self).__init__(root=root, train=train, download=download, transform=transform, target_transform=target_transform)
        # whether use generator to produce noise
        if use_generator:
            if args.gue:
                self.generator = UNet(3).cuda()
                self.generator.load_state_dict(torch.load(args.generator_filepath))
            else:
                self.noise_generator = toolbox.PerturbationTool(args,
                                        epsilon=None,
                                        num_steps=None,
                                        step_size=None)
                self.generator = self.noise_generator.generator
                ckpt = torch.load(args.generator_filepath, map_location=device)
                model_state = self.generator.state_dict()
                filtered_ckpt = {k: v for k, v in ckpt.items() 
                    if k in model_state and v.shape == model_state[k].shape}
                model_state.update(filtered_ckpt)
                self.generator.load_state_dict(model_state, strict=False)
                
            print("Generator %s loaded!" % (args.generator_filepath))
            self.generator.eval()

            self.patch_location = patch_location
            self.img_denoise = img_denoise
            self.data = self.data.astype(np.float32)

            

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise: # default: False
                targets = list(range(0, 100))
                if poison_classwise_idx is None:
                    self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                else:
                    self.poison_class = poison_classwise_idx
                self.poison_samples_idx = []
                for i, label in enumerate(self.targets):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            class_noises = {class_id: [] for class_id in range(len(self.classes))} 
            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True

                
                data_normalize = self.data[idx] / 255.0
                data_normalize_t = torch.from_numpy(np.transpose(data_normalize, (2, 0, 1))).to(device)
                # img_shape = data_normalize_t.shape
                data_input = torch.unsqueeze(data_normalize_t, 0)

                if perturb_type == 'samplewise':
                    # Sample Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    noise = noise.squeeze(0).mul(255).permute(1, 2, 0).detach().cpu().numpy()  # (32, 32, 3)
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if args.gue:
                        noise = noise * (8/255) 
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    noise = noise.squeeze(0).mul(255).permute(1, 2, 0).detach().cpu().numpy()  # (32, 32, 3)
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    class_noises[self.targets[idx]].append(noise)

            if perturb_type == 'classwise':
            
                mean_class_noises = {}
                for class_id in range(len(self.classes)):
                    if class_noises[class_id]: 
                        mean_class_noises[class_id] = np.mean(class_noises[class_id], axis=0)
                if args.gue:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                else:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                for idx in self.poison_samples_idx:
                    
                    self.data[idx] = self.data[idx] + mean_class_noises[self.targets[idx]]
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                
            self.data = self.data.astype(np.uint8)
            print('add_uniform_noise: ', add_uniform_noise)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

            
            
        else:
            
            self.perturb_tensor = torch.load(args.generator_filepath, map_location=device)
            if not isinstance(self.perturb_tensor, torch.Tensor):
                 self.perturb_tensor = torch.tensor(self.perturb_tensor)
                #  ipdb.set_trace()
                 self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
            
            if self.data.shape[2] != self.perturb_tensor.shape[2]:
                self.perturb_tensor=torch.nn.functional.interpolate(self.perturb_tensor, size=(self.data.shape[2], self.data.shape[2]), mode='bilinear', align_corners=False)
            
            print(self.perturb_tensor)
            if len(self.perturb_tensor.shape) == 4 and self.perturb_tensor.shape[3] != 3:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()
            else:
                self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 1, 3, 4, 2).to('cpu').numpy()
            self.patch_location = patch_location
            self.img_denoise = img_denoise
            self.data = self.data.astype(np.float32)
            condition = True
            if perturb_type == 'samplewise': all_len = len(self)
            else: all_len = 100  # classwise
            condition = True
            while condition:
                if len(self.perturb_tensor) < all_len:
                    # tue_len = len(self) - len(self.perturb_tensor)
                    alpha = 0.5
                    rand_noise_1 = np.random.permutation(self.perturb_tensor.shape[0])
                    rand_noise_1 = np.take(self.perturb_tensor, rand_noise_1, axis=0)
                    # ipdb.set_trace()
                    rand_noise_2 = np.random.permutation(rand_noise_1.shape[0])
                    rand_noise_2 = np.take(self.perturb_tensor, rand_noise_2, axis=0)
                    new_noise =alpha*rand_noise_1 + (1-alpha)*rand_noise_2
                    self.perturb_tensor=np.concatenate((self.perturb_tensor, new_noise), axis=0)
                    
                elif len(self.perturb_tensor) > all_len:
                    self.perturb_tensor = self.perturb_tensor[:all_len,:,:,:]
                    condition = False
                elif len(self.perturb_tensor) == all_len:
                    condition = False
            
            

            # Check Shape
            if perturb_type == 'samplewise' and self.perturb_tensor.shape[0] != len(self):
                raise('Poison Perturb Tensor size not match for samplewise')
            elif perturb_type == 'classwise' and self.perturb_tensor.shape[0] != 100:
                raise('Poison Perturb Tensor size not match for classwise')

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise:
                targets = list(range(0, 100))
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                self.poison_samples_idx = []
                for i, label in enumerate(self.targets):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = self.perturb_tensor[idx]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.perturb_tensor[self.targets[idx]]
                    noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)

                if add_uniform_noise:
                    noise = np.random.uniform(0, 8, (32, 32, 3))
                


                self.data[idx] += noise
                self.data[idx] = np.clip(self.data[idx], 0, 255)

        self.data = self.data.astype(np.uint8)
        print('add_uniform_noise: ', add_uniform_noise)
        print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))



class PoisonCIFAR101(datasets.VisionDataset):
    def __init__(self, root, split='poison_train', transform=None, target_transform=None,
                 poisn_cifar10_data=None, seed=0):
        np.random.seed(seed)
        self.transform = transform
        self.root = root
        if split == 'poison_train':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=True, download=True, transform=None)
            cifar10 = poisn_cifar10_data
            cifar10_sample_count = 500
        elif split == 'test':
            self.clean_cifar100 = datasets.CIFAR100(root=root, train=False, download=True, transform=None)
            cifar10 = datasets.CIFAR10(root=root, train=False, download=True, transform=None)
            cifar10_sample_count = 100

        self.data, self.targets = self.clean_cifar100.data, self.clean_cifar100.targets
        print(self.clean_cifar100.class_to_idx)
        # Add Ship samples of CIFAR10
        ship_idx = np.where(np.array(cifar10.targets) == 8)[0]
        selected_idx = np.random.choice(ship_idx, cifar10_sample_count, replace=False)
        extra_samples, extra_targets = [], []
        for idx in selected_idx:
            extra_samples.append(cifar10.data[idx])
            extra_targets.append(100)
        self.data = np.concatenate((self.data, np.array(extra_samples)))
        self.targets = self.targets + extra_targets
        self.poison_samples_idx = np.array(range(len(self.clean_cifar100), len(self)))
        self.poison_class = [100]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        img = Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)
        return img, target

class CustomSVHN(datasets.SVHN):
    def __getitem__(self, index):

        
        img, target = self.data[index], self.labels[index]
        
        img = transforms.ToPILImage()(img)
       
        augmented_img = Image_transform(img)
        
        original_img = Image_transform(img)
        return original_img, augmented_img, target

class PoisonSVHN(datasets.SVHN):
    def __init__(self, args, root, split='train', transform=None, target_transform=None,
                 download=False, poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, use_generator=False):
        super(PoisonSVHN, self).__init__(root=root, split=split, download=download, transform=transform, target_transform=target_transform)
        # whether use generator to produce noise
        if use_generator:
            if args.gue:
                self.generator = UNet(3).cuda()
                self.generator.load_state_dict(torch.load(args.generator_filepath))
            else:
                self.noise_generator = toolbox.PerturbationTool(args,
                                        epsilon=None,
                                        num_steps=None,
                                        step_size=None)
                self.generator = self.noise_generator.generator
                ckpt = torch.load(args.generator_filepath, map_location=device)
                model_state = self.generator.state_dict()
                filtered_ckpt = {k: v for k, v in ckpt.items() 
                    if k in model_state and v.shape == model_state[k].shape}
                model_state.update(filtered_ckpt)
                self.generator.load_state_dict(model_state, strict=False)
            print("Generator %s loaded!" % (args.generator_filepath))
            self.generator.eval()

            self.patch_location = patch_location
            self.img_denoise = img_denoise
            self.data = self.data.astype(np.float32)

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise: # default: False
                targets = list(range(0, 10))
                if poison_classwise_idx is None:
                    self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                else:
                    self.poison_class = poison_classwise_idx
                self.poison_samples_idx = []
                for i, label in enumerate(self.targets):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            class_noises = {class_id: [] for class_id in range(10)}  
            for idx in self.poison_samples_idx:
                # import ipdb; ipdb.set_trace()
                
                self.poison_samples[idx] = True

                data_normalize = self.data[idx] / 255.0
                data_normalize = torch.from_numpy(data_normalize).to(device)
                # img_shape = data_normalize.shape
                data_input = torch.unsqueeze(data_normalize, 0)
                if perturb_type == 'samplewise':
                    # Sample Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    
                    noise = noise.squeeze(0).mul(255).detach().cpu().numpy()  # (3, 32, 32)
                    if args.gue:
                        noise  = noise * (8/255) 
                    # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    
                    self.data[idx] = self.data[idx] + noise
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.generator(data_input)  # (1, 3, 32, 32)
                    
                    noise = noise.squeeze(0).mul(255).detach().cpu().numpy()  # (3, 32, 32)
                    # noise = patch_noise_extend_to_img(noise, [32, 32, 3], patch_location=self.patch_location)
                    if add_uniform_noise: # default: False
                        noise += np.random.uniform(0, 8, (32, 32, 3))
                    class_noises[self.labels[idx]].append(noise)
            if perturb_type == 'classwise':
                
                mean_class_noises = {}
                for class_id in range(10):
                    if class_noises[class_id]:  
                        mean_class_noises[class_id] = np.mean(class_noises[class_id], axis=0)
                if args.gue:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                else:
                    mean_class_noises = {k: v * (8/255) for k,v in mean_class_noises.items()}
                for idx in self.poison_samples_idx:
                    
                    self.data[idx] = self.data[idx] + mean_class_noises[self.labels[idx]]
                    self.data[idx] = np.clip(self.data[idx], a_min=0, a_max=255)

            self.data = self.data.astype(np.uint8)
            print('add_uniform_noise: ', add_uniform_noise)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

            
        else:
            self.perturb_tensor = torch.load(args.generator_filepath, map_location=device)
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).to('cpu').numpy()
            self.patch_location = patch_location
            self.img_denoise = img_denoise

            self.data = self.data.astype(np.float32)
            condition = True
            if perturb_type == 'samplewise': all_len = len(self)
            else: all_len = 10  # classwise
            condition = True
            while condition:
                if len(self.perturb_tensor) < all_len:
                    # tue_len = len(self) - len(self.perturb_tensor)
                    alpha = 0.5
                    rand_noise_1 = np.random.permutation(self.perturb_tensor.shape[0])
                    rand_noise_1 = np.take(self.perturb_tensor, rand_noise_1, axis=0)
                    # ipdb.set_trace()
                    rand_noise_2 = np.random.permutation(rand_noise_1.shape[0])
                    rand_noise_2 = np.take(self.perturb_tensor, rand_noise_2, axis=0)
                    new_noise =alpha*rand_noise_1 + (1-alpha)*rand_noise_2
                    self.perturb_tensor=np.concatenate((self.perturb_tensor, new_noise), axis=0)

                elif len(self.perturb_tensor) > all_len:
                    self.perturb_tensor = self.perturb_tensor[:all_len,:,:,:]
                    condition = False
                elif len(self.perturb_tensor) == all_len:
                    condition = False

            # Random Select Poison Targets
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            if poison_classwise:
                targets = list(range(0, 10))
                self.poison_class = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
                self.poison_samples_idx = []
                for i, label in enumerate(self.labels):
                    if label in self.poison_class:
                        self.poison_samples_idx.append(i)
            else:
                targets = list(range(0, len(self)))
                self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())

            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True
                if perturb_type == 'samplewise':
                    # Sample Wise poison
                    noise = self.perturb_tensor[idx]
                    
                elif perturb_type == 'classwise':
                    # Class Wise Poison
                    noise = self.perturb_tensor[self.labels[idx]]
                    

                if add_uniform_noise:
                    noise = np.random.uniform(0, 8, (32, 32, 3))

                self.data[idx] += noise
                self.data[idx] = np.clip(self.data[idx], 0, 255)

            self.data = self.data.astype(np.uint8)
            print('add_uniform_noise: ', add_uniform_noise)
            print(self.perturb_tensor.shape)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

import random
from collections import defaultdict

class ImageNetMini(datasets.ImageNet):
    def __init__(self, root, split='train', keep_ratio=0.2, **kwargs):
        super(ImageNetMini, self).__init__(root, split=split, **kwargs)
        if split=='train': keep_ratio=keep_ratio        
        else: keep_ratio=0.2
        self.new_targets = []
        self.new_images = []
        for i, (file, cls_id) in enumerate(self.imgs):
            if cls_id <= 99:
                self.new_targets.append(cls_id)
                self.new_images.append((file, cls_id))
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs
        class_to_imgs = defaultdict(list)
        for file, cls_id in self.imgs:
            class_to_imgs[cls_id].append((file, cls_id))
        self.new_images = []
        self.new_targets = []
        for cls_id, items in class_to_imgs.items():
            n = max(1, int(len(items) * keep_ratio))
            sampled = random.sample(items, n)
            self.new_images.extend(sampled)
            self.new_targets.extend([cls_id] * n)
        self.imgs = self.new_images
        self.targets = self.new_targets
        self.samples = self.imgs

        return
        # return

class PoisonImageNetMini(ImageNetMini):
    def __init__(self, args, root, split='train', transform=None, target_transform=None,
                 poison_rate=1.0, perturb_tensor_filepath=None,
                 seed=0, perturb_type='classwise', patch_location='center', img_denoise=False,
                 add_uniform_noise=False, poison_classwise=False, use_generator=False, class_mean_noise=None):
        super(PoisonImageNetMini, self).__init__(root=root, split=split)
        self.transform=transform
        self.target_transform=target_transform
        self.args=args
        self.poison_rate = poison_rate
        self.use_generator = use_generator
        self.add_uniform_noise = add_uniform_noise
        self.poison_classwise = poison_classwise
        self.img_denoise = img_denoise
        self.perturb_tensor_filepath = perturb_tensor_filepath
        self.seed = seed
        self.patch_location=patch_location
        self.perturb_type=perturb_type
        self.class_mean_noise = class_mean_noise

        
        if use_generator:
            np.random.seed(seed)
            self.poison_rate = poison_rate
            # Random Select Poison Targets
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True

            self.perturb_tensor = self.class_mean_noise  # (100, 224, 224, 3)
            print(self.perturb_tensor.shape)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))
        else:
            np.random.seed(seed)
            self.poison_rate = poison_rate
            self.perturb_tensor = torch.load(args.generator_filepath, map_location=device)
            self.perturb_tensor = self.perturb_tensor.mul(255).clamp_(0, 255).permute(0, 2, 3, 1).to('cpu').numpy()

            # Random Select Poison Targets
            targets = list(range(0, len(self)))
            self.poison_samples_idx = sorted(np.random.choice(targets, int(len(targets) * poison_rate), replace=False).tolist())
            self.poison_samples = collections.defaultdict(lambda: False)
            self.poison_class = []
            for idx in self.poison_samples_idx:
                self.poison_samples[idx] = True

            print(self.perturb_tensor.shape)
            print('Poison samples: %d/%d' % (len(self.poison_samples), len(self)))

    def __getitem__(self, index):
        path, target = self.samples[index]
        sample = self.loader(path)
        sample = np.array(transforms.RandomResizedCrop(224)(sample)).astype(np.float32)

        if self.poison_samples[index]:
            noise = self.perturb_tensor[target]
            sample = sample + noise               # (224, 224, 3)
            sample = np.clip(sample, 0, 255)
        sample = sample.astype(np.uint8)
        sample = Image.fromarray(sample).convert('RGB')

        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return sample, target


class Augmentation(object):
    def __init__(self, policies):
        self.policies = policies

    def __call__(self, img):
        for _ in range(1):
            policy = random.choice(self.policies)
            for name, pr, level in policy:
                if random.random() > pr:
                    continue
                img = apply_augment(img, name, level)
        return img


class CatDogDataset(datasets.VisionDataset):
    def __init__(self, root, split='train', transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.img_file_names = os.listdir(os.path.join(root, split))

    def __len__(self):
        return len(self.img_file_names)

    def __getitem__(self, index):
        filename = self.img_file_names[index]
        label = filename[:3]
        if label == 'cat':
            label = 0
        elif label == 'dog':
            label = 1
        else:
            print(filename)
            raise('Unknown label')

        with open(os.path.join(self.root, self.split, filename), 'rb') as f:
            img = Image.open(f).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            label = self.target_transform(label)

        return img, label


class CelebAMini(datasets.CelebA):
    def __init__(self, root, split="train", target_type="attr", transform=None,
                 target_transform=None, download=False, num_of_classes=1000):
        super(CelebAMini, self).__init__(root=root, split=split, target_type=target_type,
                                         transform=transform, target_transform=target_transform,
                                         download=False)

        split_map = {
            "train": 0,
            "valid": 1,
            "test": 2,
            "all": None,
        }
        split_ = split_map[datasets.utils.verify_str_arg(split.lower(), "split", ("train", "valid", "test", "all"))]

        fn = partial(os.path.join, self.root, self.base_folder)
        splits = pandas.read_csv(fn("list_eval_partition.txt"), delim_whitespace=True, header=None, index_col=0)
        identity = pandas.read_csv(fn("identity_CelebA.txt"), delim_whitespace=True, header=None, index_col=0)

        mask = slice(None) if split_ is None else (splits[1] == split_)
        identity = identity[mask]
        identity = identity[identity[1] < num_of_classes]
        self.filename = identity.index.values
        self.identity = identity.values
        print(self.identity)

    def __len__(self):
        return len(self.identity)

    def __getitem__(self, index):
        filename = self.filename[index]
        target = self.identity[index][0]
        X = Image.open(os.path.join(self.root, self.base_folder, "img_align_celeba", filename))
        if self.transform is not None:
            X = self.transform(X)
        return X, target



class Cutout(object):
    def __init__(self, length):
        self.length = length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), np.float32)
        y = np.random.randint(h)
        x = np.random.randint(w)

        y1 = np.clip(y - self.length // 2, 0, h)
        y2 = np.clip(y + self.length // 2, 0, h)
        x1 = np.clip(x - self.length // 2, 0, w)
        x2 = np.clip(x + self.length // 2, 0, w)

        mask[y1: y2, x1: x2] = 0.
        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img *= mask
        return img


class CutMix(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            bbx1, bby1, bbx2, bby2 = rand_bbox(img.size(), lam)
            img[:, bbx1:bbx2, bby1:bby2] = img2[:, bbx1:bbx2, bby1:bby2]
            lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (img.size()[-1] * img.size()[-2]))
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)


class MixUp(Dataset):
    def __init__(self, dataset, num_class, num_mix=2, beta=1.0, prob=0.5):
        self.dataset = dataset
        self.num_class = num_class
        self.num_mix = num_mix
        self.beta = beta
        self.prob = prob

    def __getitem__(self, index):
        img, lb = self.dataset[index]
        lb_onehot = onehot(self.num_class, lb)

        for _ in range(self.num_mix):
            r = np.random.rand(1)
            if self.beta <= 0 or r > self.prob:
                continue

            # generate mixed sample
            lam = np.random.beta(self.beta, self.beta)
            rand_index = random.choice(range(len(self)))

            img2, lb2 = self.dataset[rand_index]
            lb2_onehot = onehot(self.num_class, lb2)

            img = img * lam + img2 * (1-lam)
            lb_onehot = lb_onehot * lam + lb2_onehot * (1. - lam)

        return img, lb_onehot

    def __len__(self):
        return len(self.dataset)

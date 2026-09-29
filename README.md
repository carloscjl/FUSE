# FUSE: Full‑spectrum Unlearnable Examples via Spectral Equalization
Abstract: Unlearnable examples (UEs) protect training data by injecting imperceptible perturbations so that models fail to extract exploitable representations. In this paper, we reveal that existing UEs exhibit a critical failure once low-pass filtering is applied, indicating that the effective perturbation signals for unlearnability concentrate predominantly in high frequencies. Hence, we argue that reliable UEs should remain effective across the full spectrum. To this end, we propose Full-spectrum Unlearnable Examples via Spectral Equalization (FUSE), which aims to generate spectrum-agnostic perturbations by equalizing the contributions from different bands and enforcing cross-band consistency. Specifically, FUSE adopts a Random Spectral Masking (RSM) strategy during generator training, which randomly removes a contiguous frequency band, forcing the remaining bands to maintain unlearnability. In addition, FUSE further integrates Cross-Band Guidance (CBG), which enforces mutual consistency between high- and low-frequency components, thereby further enhancing low-frequency unlearnability and regulating high-frequency perturbations to preserve the semantic fidelity of images. Extensive experiments across multiple datasets, architectures, and spectral filtering demonstrate the strong protection achieved by FUSE.

## Citation

If you find this work useful in your research, please consider citing:

```bibtex
@inproceedings{cai2026fuse,
  title={FUSE: Full-spectrum Unlearnable Examples via Spectral Equalization},
  author={Cai, Jiale and Xu, Gezheng and Li, Zhihao and Fang, Ruiyi and Pu, Ruizhi and Wu, Di and Lao, Qicheng and Ling, Charles and Wang, Boyu},
  booktitle={Forty-third International Conference on Machine Learning},
  year={2026}
}
```

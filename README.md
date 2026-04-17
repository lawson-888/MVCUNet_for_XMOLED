# MVCUNet for XMOLED

This repository contains the official implementation of our work on quantitative MR parameters reconstruction based on MVCUNet.

---


## Project Structure

```text
MVCUNet_for_XMOLED/
│
├── MVCUnet_model.py      # MVCUNet architecture
├── mamba_model.py        # Mamba-based modules
├── vmamba.py             # Vision Mamba implementation
├── train_MVCUnet.py      # Training script
├── requirements.txt      # Dependencies
└── README.md
```

---


## Environment

- Python 3.8.20
- PyTorch 1.13.0
- NVIDIA GeForce RTX 4090 GPU
- Ubuntu 16.04.7 LTS 

---

## Data Availability

The numerical phantom templates are provided. https://pan.baidu.com/s/1l_bqKNc2sORllqIkqVpZHw?pwd=1234

Due to privacy and ethical restrictions, human brain data may be made available from the authors upon reasonable request and subject to institutional approval.

---


## References and Acknowledgements

Parts of this code are adapted from the following open-source projects:

https://github.com/johnma2006/mamba-minimal
Licensed under the Apache License 2.0

https://github.com/JCruan519/VM-UNet
License: Apache License 2.0

We gratefully acknowledge the original authors for making their code publicly available.


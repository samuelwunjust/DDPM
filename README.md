# DDPM
The simple implenmentation of DDPM
cuda1.11.0
RTX3090
This model use *Cifar10 datasets*:https://www.kaggle.com/datasets/joaopauloschuler/cifar10-64x64-resized-via-cai-super-resolution
*checkpoints*:https://www.kaggle.com/datasets/samuelngnjust/ddpm-checkpoint-ema

The *structure* of network is *Unet_conditional* which is:
"""Unet结构Input
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
        DoubleConv
           |
        Down -> SelfAttention
           |
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
        DoubleConv
           |
        Up -> SelfAttention
           |
           |
        Output"""

        
Using"!python DDPM_conditional.py"to train




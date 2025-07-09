# LDANet

## Introduction

This is our implementation of our paper *Lightweight Deformable Attention for Event-base Monocular Depth Estiomation.* Authors: Shaofan Wang, Yanfeng Sun, Baocai Yin. Submitted to Displays.

**Abstract:**

Event cameras are neuromorphically inspired sensors that output brightness changes in the form of a stream of asynchronous events instead of  intensity frames. Event-based monocular depth estimation forms a foundation of widespread high dynamic vision applications. Existing monocular depth estimation networks, such as CNNs and transformers, suffer from the insufficient exploration of spatiotemporal correlation, and the high complexity. In this paper, we propose the Lightweight Deformable Attention Network (LDANet) for circumventing the two issues. The key component of LDANet is  Mixed Attention with Temporal Embedding module, which consists of a lightweight deformable attention layer and a temporal embedding layer. The former, as an improvement of deformable attention, is equipped with a drifted token representation and a K-nearest multi-head deformable-attention block, capturing the locally-spatial correlation. The latter is equipped with a cross-attention layer by querying the previous temporal event frame, encouraging to memorize the history of depth clues and capturing temporal correlation. Experiments on a real scenario dataset and a simulation scenario dataset show that, LDANet achieves a satisfactory balance between the inference efficiency and depth estimation accuracy. 

## Dependencies

- Python == 3.9
- PyTorch == 2.4.1

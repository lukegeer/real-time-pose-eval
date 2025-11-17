# real-time-pose-eval
Evaluating the Limits of Real-Time Complex Pose Estimation and Reference Matching

## Authors
Lukas Geer - lgeer3@jh.edu  
Max-Peter Schrøder - mschrod2@jh.edu  
Valerie Liang - vliang5@jh.edu  
Maaz Shamim - mshamim2@jh.edu  

## Overview
A lightweight hybrid computer vision based live video processing pipeline capable of real-time comparison of body poses to a reference pose, estimating and scoring the posing accuracy between input and reference, and ensuring robustness to temporal and physical disturbances.

## Problem
While body tracking has become increasingly common in gaming and motion-based applications, common challenges such as occlusion, temporal jitter, and latency may lead to unstable detections, causing the models to be unstable in continuous interactions like dynamic activities. Furthermore, mapping the current input pose to a reference presents challenges such as body proportion scaling and posing accuracy. 

Our project will evaluate and attempt to improve these methods by directly comparing state-of-the-art body tracking models with our algorithms in scenarios that demand precise and efficient body alignment to estimate the position of obscured or jittered body parts frame-by-frame. We will focus the training, testing, and evaluation of our models to a restricted subspace of poses, targeting motions that require precise body mapping and timing.

## Data
We will use the [AIST++](https://google.github.io/aistplusplus_dataset/factsfigures.html) dataset for training and evaluation, which offers over 10 million annotated frames of 3D dance motion across ten genres. This provides the complex, dynamic poses necessary to rigorously test pose estimation.We will use a single-camera view to focus the challenge on accurately predicting occluded joints without multi-view cues. The dataset's video structure allows us to test both static frame accuracy (for handling self-occlusion) and temporal performance (for consistency and latency).

## Methodology

## Architecture

## Installation

## Results

## Sources

- [Optical flow source code](https://github.com/iv4n-ga6l/Optical_flow_opencv/tree/main)


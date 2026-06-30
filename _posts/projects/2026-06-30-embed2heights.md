---
layout: post
title: Dense prediction from frozen geospatial foundation-model embeddings
image_show_on_page: false
image: assets/images/project/embed2heights.png
category: project
description: ESA Φ-lab "embed2heights" GeoAI challenge — segmenting buildings, vegetation and water and regressing height from frozen GFM embeddings, and what it taught me about evaluation
permalink: embed2heights
---
Entry and empirical study for the ESA Φ-lab "Reaching new heights with GeoFM" challenge (team ImageX). The task: from six pre-computed, frozen geospatial foundation-model embeddings over 256×256 tiles at 10 m, predict per-pixel building/vegetation/water cover and an above-ground height (nDSM). Only a lightweight decoder is trained; the foundation models stay frozen.<br>
<br>
The embeddings span two spatial scales: dense 10 m pixel embeddings (AlphaEarth, 64-d; TESSERA, 128-d) and coarse 16×16 patch tokens at about 160 m from TerraMind and THOR (Sentinel-1 SAR and Sentinel-2 optical, 768-d each). I tried fusing the dense sources by per-source projection and concatenation, bringing in the coarse tokens by learned upsampling and by cross-attention, a resolution-preserving dilated decoder, sub-pixel super-resolution heads, a class-wise height loss in linear metres to match the RMSE metric, augmentation, test-time augmentation, and small ensembles.<br>

<h4>The honest result: simple won</h4>
The most elaborate configuration scored the worst, and the simplest one scored the best. A plain two-embedding (AlphaEarth + TESSERA) U-Net reached the highest leaderboard score (0.398). Adding the other four embeddings, fancier fusion, augmentation, an ensemble, and test-time augmentation steadily lowered it (down to 0.371). Vegetation and water segmentation and the height regression were solid throughout; building footprints were the bottleneck and never moved.<br>
<br>
Building footprints hit a hard ceiling near 0.33 IoU that no architecture, fusion scheme, super-resolution head, or loss change could shift. At 10 m a building is one to three pixels at under 1% prevalence, so near-identical embeddings carry different labels and the mapping is not learnable past that point. Even the training-set fit caps around 0.46, so this is a property of the input resolution, not of the model.<br>

<h4>The part I found most useful: evaluation didn't transfer</h4>
The lasting lesson was about model selection. I compared three signals and they disagreed. A random train/validation split inflated scores and mis-ranked models, because neighbouring tiles leak across the split through spatial autocorrelation. Moving to a region-grouped split, holding out whole geographic tiles, fixed that pathology in head-to-head comparisons. But when I trained the region-CV-preferred recipe on all the data and submitted it, it still regressed on the real test set. So neither random nor spatial cross-validation reliably predicted the held-out test ranking; only the leaderboard did, and the simplest baseline was the most robust.<br>
<br>
The practical takeaway for anyone building on these embeddings: validate by region rather than by random tile, but do not over-trust any local proxy on this kind of task, and keep a simple baseline as the reference you have to beat on the real test, not on a held-out fold.<br>

<h4>Takeaway</h4>
Frozen 10 m GFM embeddings with a trained decoder are strong for area-like classes such as vegetation and water and for height, but bounded for sub-pixel structure like building footprints, and elaboration past a simple baseline did not help here. Breaking the footprint ceiling needs richer input, higher-resolution imagery or learned feature super-resolution of the embeddings, rather than a larger decoder.<br>

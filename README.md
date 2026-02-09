---
title: Traffic Sign Recognition AI
emoji: 🚦
colorFrom: blue
colorTo: purple
sdk: docker
pinned: false
license: mit
---

# 🚦 Traffic Sign Recognition AI

A deep learning application for real-time traffic sign classification using a CNN trained on the German Traffic Sign Recognition Benchmark (GTSRB).

## Features

- **99.7% Accuracy** - High performance on 43 traffic sign classes
- **Real-time Predictions** - Instant results in under 1 second
- **Interactive UI** - Beautiful, spacious interface built with Streamlit
- **Multiple Formats** - Supports PNG, JPG, JPEG, and WebP images

## Model

- **Architecture**: CNN with 3 convolutional blocks, BatchNormalization, and Dropout
- **Parameters**: 1.3 million
- **Training Data**: 39,209 images from GTSRB dataset
- **Input Size**: 48×48 pixels RGB

## Usage

Upload any traffic sign image and get instant predictions with confidence scores and top-5 alternatives.

## License

MIT License

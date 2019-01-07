# pytorch-conv4d

This repository contains a simple PyTorch port of the [conv4d for TensorFlow repository](https://github.com/funkey/conv4d) by Jan Funke. It consists essentially of a single class, `Conv4d`, which provides a (still rather rudimentary) PyTorch layer for 4-dimensional convolutions. Like the original, it works by performing and stacking several 3D convolutions (see the original repository for a more detailed explanations).

This implementation is still work in progress (hence it comes with no warranties whatsoever), and pull requests or advice for improvements are very much welcome! :)
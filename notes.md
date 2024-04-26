### Training Process

Train the model (or possibly a smaller version) on a holdout set (the authors split the training set into holdout and training), and save the irreducible losses over the training set in a dataloader.

TODO: Write generators and irreducible training script. Also, do not need to shove the irreducible loss into a dataloader, can simply pass around the model.

### Our contibutions

- Adapted the implementation to Tensorflow, not as refined as the original submission, but we have no plans of going to CVPR anytime soon.
- Adaptation to Audio modality -> even though spectrograms look and feel like images, they can be very subtly noisy.
- Datasets used : urban_sounds_small, audio-mnist
- Models : Both spectrogram-based and temporal models are evaluated.
- Selection Methods: reducible loss selection, autoencoder-based selection, uniform sampling

On an unrelated but interesting note,

- As a side effect, we introduce attention masking on spectrograms and obtain better latent representations.

### Citations

@article{audiomnist2023,
title = {AudioMNIST: Exploring Explainable Artificial Intelligence for audio analysis on a simple benchmark},
journal = {Journal of the Franklin Institute},
year = {2023},
issn = {0016-0032},
doi = {https://doi.org/10.1016/j.jfranklin.2023.11.038},
url = {https://www.sciencedirect.com/science/article/pii/S0016003223007536},
author = {Sören Becker and Johanna Vielhaben and Marcel Ackermann and Klaus-Robert Müller and Sebastian Lapuschkin and Wojciech Samek},
keywords = {Deep learning, Neural networks, Interpretability, Explainable artificial intelligence, Audio classification, Speech recognition},
}

# Third-party attribution

This repository contains my experiment scripts, run configurations, measurement records, and curated visual results. Full neural network implementations, pretrained checkpoints, and source benchmark archives are not included. Small compatibility patches retain references to the corresponding upstream code.

- **SAFMN** by its original authors: https://github.com/sunny2109/SAFMN. The training and test YAMLs here are adaptations of its BasicSR configuration format and retain the upstream model names and hyperparameters. The upstream repository distributes its code under Apache License 2.0; a copy is in [`third_party/Apache-2.0.txt`](third_party/Apache-2.0.txt).
- **CAMixerSR** by its original authors: https://github.com/icandle/CAMixerSR. Its model and original pretrained checkpoint are linked, not redistributed here. Test/training YAMLs and the compatibility patch adapt its BasicSR extension for the recorded runs. The upstream repository includes an Apache License 2.0 file.
- **BasicSR**: https://github.com/XPixelGroup/BasicSR. Used through the model repositories and studied separately; its source tree is not included.
- **DIV2K**: https://data.vision.ee.ethz.ch/cvl/DIV2K/. The official page describes academic research use and requests citation. No images or archives are included.
- **Flickr2K**: https://cv.snu.ac.kr/research/EDSR/. No images or archives are included.
- **Set5 mirror**: https://huggingface.co/datasets/eugenesiow/Set5. Used for five-image protocol checks. Curated network predictions and comparison figures, including ground-truth panels, are derived from those benchmark images. They illustrate measured experiments and are not original photographs by the repository owner. Standalone source HR/LR images and dataset archives are not redistributed.

The public repository does not include the model checkpoints trained in this experiment. Metrics and hashes are provided so the local records can be matched to the original artifacts.

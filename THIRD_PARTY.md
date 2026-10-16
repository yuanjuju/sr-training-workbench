# Third-party attribution

This repository contains my experiment scripts, run configurations, and measurement records. The neural network implementations, pretrained checkpoints, and benchmark datasets are not included.

- **SAFMN** by its original authors: https://github.com/sunny2109/SAFMN. The training and test YAMLs here are adaptations of its BasicSR configuration format and retain the upstream model names and hyperparameters. The upstream repository distributes its code under Apache License 2.0; a copy is in [`third_party/Apache-2.0.txt`](third_party/Apache-2.0.txt).
- **CAMixerSR** by its original authors: https://github.com/icandle/CAMixerSR. Its model and original pretrained checkpoint are linked, not redistributed here. The test YAML is an adaptation for the local Set5 check. The upstream repository includes an Apache License 2.0 file.
- **BasicSR**: https://github.com/XPixelGroup/BasicSR. Used through the model repositories and studied separately; its source tree is not included.
- **DIV2K**: https://data.vision.ee.ethz.ch/cvl/DIV2K/. The official page describes academic research use and requests citation. No images or archives are included.
- **Flickr2K**: https://cv.snu.ac.kr/research/EDSR/. No images or archives are included.
- **Set5 mirror**: https://huggingface.co/datasets/eugenesiow/Set5. Used for a five-image protocol check; no images or derived visual examples are included.

The public repository does not include the model checkpoints trained in this experiment. Metrics and hashes are provided so the local records can be matched to the original artifacts.

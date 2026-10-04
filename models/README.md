# Additional saved models

Copy the contents of your Google Drive **My Models** folder here, preserving filenames. The files below were visible in the supplied folder screenshot. They have not been supplied as model binaries, and are not included in this repository snapshot. Sizes are approximate values displayed in the screenshot.

| Filename | Displayed size | Status |
| --- | ---: | --- |
| `AlexNet.h5` | 810.8 MB | To be added |
| `AlexNet.keras` | 810.8 MB | To be added |
| `DenseNet.keras` | 113.8 MB | To be added |
| `Multi-Head_U-Net.keras` | 3.6 MB | To be added |
| `Multi-Head_U-Net2.h5.npy` | 3.5 MB | To be added |
| `Multi-Head_U-Net2.keras` | 3.5 MB | To be added |
| `ResNet_resaved.keras` | 1.3 MB | To be added |
| `ResNet.h5` | 3.7 MB | To be added |
| `ResNet.keras` | 1.3 MB | To be added |
| `VGG16.keras` | 116.3 MB | To be added |

The separately supplied `../resnet-standalone/ResNet.keras` is approximately 3.9 MB (decimal bytes rounded), with Keras 2.15 metadata. The screenshot's `ResNet.keras` is displayed as 1.3 MB. Keep them separate until their versions and provenance are checked; identical names do not establish identical checkpoints.

The `.h5.npy` filename does not establish whether that file is a complete model or a NumPy object/weights export. Check its actual format before selecting a loader. U-Net checkpoints can require notebook-defined custom loss functions for compiled loading.

## Add through Git LFS

The root `.gitattributes` already marks model files in this folder for Git LFS. Install Git LFS and run `git lfs install` before staging the models. From the repository root:

```bash
git add .gitattributes models/
git lfs status
```

Confirm the large weights appear in the LFS status output before committing and pushing. GitHub's browser upload interface cannot perform this conversion for you. See the [main README](../README.md#adding-the-saved-model-folder-to-github) for the complete initial repository setup and [GitHub's file-size limits](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

When adding the actual files, update this inventory with their status, checksums, input/output shapes, framework versions, and associated notebook/training run. Some notebooks still reference Google Drive paths; update those paths explicitly before using the local models.

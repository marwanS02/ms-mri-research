# Multiple Sclerosis Lesion Detection in MRI

Research notebooks for detecting lesion presence in FLAIR and T2-weighted brain MRI slices, comparing convolutional neural networks, and exploring joint segmentation and classification. This repository collects the software experiments underlying two research papers and an expanded ResNet analysis notebook with a saved model.

The main classification task is **slice-level lesion presence**: label `1` means the corresponding manual lesion mask contains at least one lesion pixel; label `0` means it does not. A lesion-negative slice is not a healthy control subject. The source dataset contains patients with MS.

## Associated papers

1. **Deep Learning Approaches for Multiple Sclerosis Detection in MRI Images** — Mohamad Marwan El Sidani, Rita Younes, Charles Yaacoub, and Roy Abi Zeid Daou. *BioMed Research International*, 2026, article 5726771. [DOI: 10.1155/bmri/5726771](https://doi.org/10.1155/bmri/5726771) · [Included PDF](papers/deep-learning-ms-detection.pdf).
2. **Detection of Multiple Sclerosis Based on MRI Images using a Low-Power Analog Integrated Artificial Neural Networks** — Vassilis Alimisis, Mohamad Marwan El Sidani, Rita Younes, Konstantinos Cheliotis, Paul P. Sotiriadis, Jalal Possik, Charles Yaacoub, and Roy Abi Zeid Daou. [Included manuscript](papers/analog-ms-classification.pdf). Publication venue, year, and DOI are not specified in the supplied manuscript.

The first paper compares AlexNet-C, pretrained VGG16, ResNet-10, and DenseNet-121. The second investigates software-trained networks mapped to low-power analog hardware. This repository contains the supplied MRI software notebooks and model artifact; analog circuit netlists, layout files, hardware simulations, and a dedicated MLP hardware-mapping pipeline are not included. The U-Net notebooks are exploratory work, rather than an additional benchmark reported in the CNN comparison paper.

## Repository contents

| Path | Contents |
| --- | --- |
| [`notebooks/AlexNet.ipynb`](notebooks/AlexNet.ipynb) | Custom AlexNet-style classifier, preprocessing, augmentation, training, and evaluation. |
| [`notebooks/VGG16.ipynb`](notebooks/VGG16.ipynb) | VGG16 classifier experiments, including an ImageNet-pretrained backbone and grayscale-to-RGB conversion. |
| [`notebooks/ResNet.ipynb`](notebooks/ResNet.ipynb) | Custom residual-network classifier experiments, with multiple architecture and training variants. |
| [`notebooks/DenseNet.ipynb`](notebooks/DenseNet.ipynb) | Custom DenseNet classifier with dense-block configuration `[6, 12, 24, 16]`. |
| [`notebooks/Multi-Head U-Net.ipynb`](<notebooks/Multi-Head U-Net.ipynb>) | Experimental shared encoder with segmentation and classification outputs; several loss and training variants. |
| [`notebooks/Multi-Head U-Net version 2.ipynb`](<notebooks/Multi-Head U-Net version 2.ipynb>) | Further joint segmentation/classification and training experiments. |
| [`notebooks/MS_Classification.ipynb`](notebooks/MS_Classification.ipynb) | Early exploratory notebook with multiple architectures and mixed TensorFlow/PyTorch experiments. |
| [`resnet-standalone/`](resnet-standalone/) | Expanded `ResNet(1).ipynb`, supplied `ResNet.keras`, model documentation, and a small smoke-check script. |
| [`models/`](models/) | Location for the additional saved models from the original Google Drive **My Models** folder; see its inventory and Git LFS instructions. |
| [`papers/`](papers/) | Copies of the two supplied papers under shorter filenames. |

Notebook source cells and saved outputs are preserved. Dataset volumes, cached arrays, external-validation DICOM images, and manual-label CSV files are not bundled. The additional model files listed in `models/README.md` still need to be copied into that folder.

## Dataset and preprocessing

The experiments use [Brain MRI Dataset of Multiple Sclerosis with Consensus Manual Lesion Segmentation and Patient Meta Information](https://data.mendeley.com/datasets/8bctsm8jz7/1), by Ali M Muslim, version 1, 2022. [Dataset DOI: 10.17632/8bctsm8jz7.1](https://doi.org/10.17632/8bctsm8jz7.1). The dataset provides multisequence MRI and manual lesion masks for 60 MS patients. The CNN paper reports 2,831 FLAIR/T2 slices in the compiled experiments. Download the dataset separately and retain its attribution and license information.

The classification loaders expect this naming pattern beneath a configurable `base_path`:

| Example path | Meaning |
| --- | --- |
| `Patient-1/1-Flair.nii` | FLAIR volume. |
| `Patient-1/1-T2.nii` | T2 volume. |
| `Patient-1/1-LesionSeg-Flair.nii` | FLAIR lesion mask. |
| `Patient-1/1-LesionSeg-T2.nii` | T2 lesion mask. |

The pattern repeats for `Patient-2` through `Patient-60`. Adapt the filenames if your dataset extraction differs. The supplied loaders use uncompressed `.nii` filenames.

In the main classification notebooks, each axial slice is divided by its maximum intensity when that maximum is positive, resized to `300 × 300` with anti-aliasing, and zero-padded to `320 × 320`. A channel dimension is added for grayscale input. VGG16 uses three channels derived from grayscale. FLAIR and T2 slices are appended as separate samples, rather than stacked as two input channels.

Classification arrays are cached as `X.npy` and `Y.npy`; the joint U-Net experiments use `X.npy`, `Y_seg.npy`, and `Y_class.npy`. Check each notebook's save and load paths, because some cells write to different cache directories.

The principal classification notebooks use two `train_test_split` calls with `random_state=42`: 20% is held out for testing, then 25% of the remaining 80% for validation, giving approximately **60% training / 20% validation / 20% testing**. These are slice-level splits and do not enforce patient separation. The analog manuscript describes a different 70/30 protocol; that protocol should not be inferred from these CNN split cells.

## Getting started

### 1. Set up a compatible environment

The notebooks were developed in Google Colab and use Google Drive paths. For the original Keras 2 code and supplied model, use a separate **Python 3.10 or 3.11** environment:

```bash
python -m venv .venv
```

Activate it with `source .venv/bin/activate` on Linux/macOS or `.venv\Scripts\Activate.ps1` in Windows PowerShell, then install the baseline dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
jupyter lab
```

`requirements.txt` provides a Keras 2.15 baseline aligned with the supplied model's metadata, rather than a complete lockfile of the historical training environment. TensorFlow 2.15 installs Keras 2.15; TensorFlow 2.16 and later use Keras 3 by default. See [Keras compatibility guidance](https://keras.io/getting_started/).

The expanded `resnet-standalone/ResNet(1).ipynb` installs TensorFlow 2.19.1 in its first cell and also retains legacy imports. For the Keras 2 baseline, skip that installation cell. To work with its newer environment, adapt the legacy imports/APIs and verify model loading separately. The notebook is not guaranteed to run unchanged under both Keras versions.

Optional dependencies: install `torch` and `tqdm` only if running the PyTorch cells in `MS_Classification.ipynb`. Architecture plotting may require the Graphviz system executable in addition to `pydot`. TeX-based figure formatting is optional; disable `text.usetex` if LaTeX is unavailable.

### 2. Configure data and model paths

- **Colab:** mount your Google Drive and update `base_path`, `base_save_path`, and all explicit load/save paths to your folders. Install the dependencies in the active runtime and restart it after changing TensorFlow/Keras versions.
- **Local Jupyter:** remove or skip `from google.colab import drive` and `drive.mount(...)`, then replace `/content/drive/...` paths with local paths. Do not install `google.colab` merely to run the local notebooks.
- **Expanded ResNet analysis:** configure DICOM directories, mask directories, and label CSV paths only when running those optional sections; these files are not included.

### 3. Choose and run an experiment

Open one architecture notebook, load the dataset or its cached arrays, choose the intended model definition and training block, and then run its evaluation cells. Review alternative blocks before execution: later definitions can overwrite earlier models, and some cells retrain or overwrite saved files. `MS_Classification.ipynb` and the U-Net notebooks are exploratory archives and include unfinished or alternative cells; they are not automatic end-to-end pipelines.

To inspect the provided model without installing TensorFlow or obtaining the dataset:

```bash
python resnet-standalone/smoke_check.py
```

For an actual load and synthetic forward pass in the Keras 2 environment:

```bash
python resnet-standalone/smoke_check.py --predict
```

See the [standalone ResNet README](resnet-standalone/README.md) for model metadata and a minimal inference example.

## Results reported in the CNN paper

The following values are transcribed from Tables 3 and 5 of the BioMed Research International paper. They have not been recomputed from this repository snapshot.

| Model | ROC AUC | Precision | Recall | F1-score | Accuracy |
| --- | ---: | ---: | ---: | ---: | ---: |
| AlexNet-C | 0.90 | 0.84 | 0.84 | 0.84 | 0.84 |
| Pretrained VGG16 | 0.94 | 0.81 | 0.93 | 0.86 | 0.85 |
| ResNet-10 | 0.90 | 0.79 | 0.94 | 0.86 | 0.84 |
| DenseNet-121 | 0.91 | 0.82 | 0.84 | 0.83 | 0.83 |

The supplied `.keras` file is not proven to be the exact checkpoint used for every reported ResNet result. The analog paper's hardware results require the separate circuit workflow and cannot be reproduced from these notebooks alone.

## Reproducibility notes

- Preserve patient identifiers when creating new splits. Correlated slices or different modalities from the same patient can occur in multiple subsets under the existing slice-level protocol; these results should not be treated as patient-independent evaluation.
- Record the selected architecture variant, checkpoint, dependency versions, data ordering, preprocessing, seed, and threshold for each experiment. Multiple definitions and retained outputs do not constitute an exact experiment manifest.
- The supplied ResNet checkpoint has a single sigmoid output after global average pooling. Other notebook variants add a 512-unit dense layer; they are different architectures.
- Grad-CAM heatmaps and overlap proxies in the expanded notebook are interpretability analyses, not outputs from a trained lesion-segmentation model.
- Training and external-validation execution require data that is not bundled. A successful synthetic forward pass checks loading and dimensions, not predictive accuracy.

## Adding the saved-model folder to GitHub

Copy the contents of your Google Drive **My Models** folder into [`models/`](models/). Several models shown in the folder inventory exceed GitHub's regular Git file limit. The included `.gitattributes` tracks `.keras`, `.h5`, and `.h5.npy` files inside `models/` with Git LFS. The small checkpoint in `resnet-standalone/` remains a regular Git file.

Install [Git LFS](https://git-lfs.com/) and, from the extracted repository folder, run these commands **before the first commit**:

```bash
git init
git lfs install
git add .
git lfs status
git commit -m "Add MRI research notebooks and documentation"
```

Create an empty GitHub repository, then connect its actual URL and push using Git or GitHub Desktop. GitHub browser uploads allow up to 25 MiB per file; regular Git blocks files larger than 100 MiB. Adding `.gitattributes` through the browser does not convert uploaded weights into LFS objects. See [GitHub's large-file documentation](https://docs.github.com/en/repositories/working-with-files/managing-large-files/about-large-files-on-github).

After cloning a repository that includes LFS models, run `git lfs pull` if the weights are still pointer files. An external model download location is also an option; document its file versions and checksums if used.

## Citation

If you use these experiments, cite the associated paper relevant to your work and the original dataset. A BibTeX entry for the published CNN paper is included below. For the analog manuscript, use the author list and title above until final publication details are available.

```bibtex
@article{elsidani2026ms,
  author  = {El Sidani, Mohamad Marwan and Younes, Rita and
             Yaacoub, Charles and Abi Zeid Daou, Roy},
  title   = {Deep Learning Approaches for Multiple Sclerosis
             Detection in MRI Images},
  journal = {BioMed Research International},
  year    = {2026},
  volume  = {2026},
  pages   = {5726771},
  doi     = {10.1155/bmri/5726771}
}
```

## License and use

No software or model license has been assigned in this repository snapshot. The included articles and the external dataset retain their own licenses; the published article's license does not automatically license the source code or weights. These artifacts are research experiments and have not been validated here for clinical use.

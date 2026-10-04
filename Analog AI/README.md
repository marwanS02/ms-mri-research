# Standalone ResNet notebook and checkpoint

This folder keeps the expanded [`ResNet(1).ipynb`](<ResNet(1).ipynb>) alongside the supplied [`ResNet.keras`](ResNet.keras). The notebook includes training variants, Grad-CAM visualization, calibration and confidence analysis, DICOM browsing, manual labeling, and external-data evaluation. The data used by these sections is not included.

## Supplied checkpoint

The following details come directly from the `.keras` archive:

| Property | Value |
| --- | --- |
| Saved Keras version | `2.15.0` |
| Save timestamp | `2024-05-01@01:33:00` (timezone unspecified) |
| Model type | Keras functional model, custom residual CNN |
| Input | `(batch, 320, 320, 1)` grayscale slices |
| Residual stages | 16, 32, 64, and 128 filters |
| Classifier | Global average pooling followed by one sigmoid unit |
| Additional 512-unit dense layer | Absent from this saved checkpoint |
| Output interpretation | Lesion-positive slice score, following the notebooks' label convention |

The archive contains `metadata.json`, `config.json`, and `model.weights.h5`. Its provenance has not been matched to an exact training run or every result in the papers. Other files named `ResNet.keras` in your model collection may be different versions.

SHA-256 of the supplied checkpoint: `c735ee81c3f3d88aec812f95315e9656d1f2557256e09e6ca54887ef1fd38493`.

## Quick sanity check

From this folder, inspect archive integrity, configuration, weight-file signature, and SHA-256 using only the Python standard library:

```bash
python smoke_check.py
```

For a load and synthetic forward pass, use Python 3.10 or 3.11 in an activated virtual environment:

```bash
python -m pip install -r requirements.txt
python smoke_check.py --predict
```

The script locates `ResNet.keras` relative to itself, so it also works when invoked from the repository root. The prediction mode checks a finite `(1, 1)` sigmoid result on synthetic input; it does not measure accuracy.

## Minimal inference example

Run from this folder after installing its requirements:

```python
import numpy as np
from tensorflow.keras.models import load_model

model = load_model("ResNet.keras", compile=False)
# Synthetic input only; replace with slices preprocessed like the training data.
x = np.zeros((1, 320, 320, 1), dtype=np.float32)
scores = model.predict(x, verbose=0)
print("Lesion-positive slice score:", float(scores[0, 0]))
```

For real input, follow the training preprocessing: maximum-intensity normalization, resizing to `300 × 300`, padding to `320 × 320`, and adding the channel and batch dimensions. A threshold of `0.5` is a simple example; use the threshold appropriate to the particular validated experiment. The output is not a patient-level diagnosis.

## Running the notebook

Use `jupyter lab` from the root environment, or open the notebook in Colab. Replace its Google Drive data and model paths; use `ResNet.keras` for local loading. If you only need inference, use the example or smoke check without executing training cells.

The notebook's first cell installs TensorFlow 2.19.1, whereas the supplied checkpoint was saved with Keras 2.15. Skip that cell when using these baseline requirements. The retained legacy `keras.preprocessing.image` imports and other version-specific APIs may need adaptation under Keras 3. Multiple model definitions and training blocks are retained, so choose the intended variant instead of running every cell blindly. See the [main README](../README.md) for dataset setup, paper references, and reproducibility notes.

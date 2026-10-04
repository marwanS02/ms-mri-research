"""Inspect the supplied model; optionally load it for a synthetic forward pass."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import zipfile


def inspect_archive(path: Path) -> dict:
    """Validate the expected archive structure and return its saved metadata."""
    with zipfile.ZipFile(path) as archive:
        required = {"metadata.json", "config.json", "model.weights.h5"}
        missing = required.difference(archive.namelist())
        if missing:
            raise ValueError(f"Missing archive members: {sorted(missing)}")
        corrupt = archive.testzip()
        if corrupt is not None:
            raise ValueError(f"Archive CRC check failed: {corrupt}")
        metadata = json.loads(archive.read("metadata.json"))
        config = json.loads(archive.read("config.json"))
        layers = config["config"]["layers"]
        inputs = [layer for layer in layers if layer["class_name"] == "InputLayer"]
        if len(inputs) != 1:
            raise ValueError("Expected one input layer")
        shape = inputs[0]["config"].get("batch_input_shape")
        if shape != [None, 320, 320, 1]:
            raise ValueError(f"Unexpected input shape: {shape}")
        output = layers[-1]
        if (output["class_name"] != "Dense"
                or output["config"].get("units") != 1
                or output["config"].get("activation") != "sigmoid"):
            raise ValueError("Expected one sigmoid output unit")
        with archive.open("model.weights.h5") as weights:
            if weights.read(8) != b"\x89HDF\r\n\x1a\n":
                raise ValueError("Weights do not have the expected HDF5 signature")
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return {"keras_version": metadata.get("keras_version"),
            "date_saved": metadata.get("date_saved"),
            "input_shape": shape, "sha256": digest.hexdigest()}


def check_prediction(path: Path) -> float:
    """Check loading and output dimensions; this is not an accuracy test."""
    import numpy as np
    from tensorflow.keras.models import load_model

    model = load_model(path, compile=False)
    sample = np.zeros((1, 320, 320, 1), dtype=np.float32)
    result = np.asarray(model(sample, training=False))
    if result.shape != (1, 1):
        raise ValueError(f"Unexpected prediction shape: {result.shape}")
    if not np.isfinite(result).all() or not ((result >= 0) & (result <= 1)).all():
        raise ValueError("Prediction is not a finite sigmoid score")
    return float(result[0, 0])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path,
                        default=Path(__file__).resolve().with_name("ResNet.keras"))
    parser.add_argument("--predict", action="store_true",
                        help="Also load with TensorFlow and run synthetic input")
    args = parser.parse_args()
    try:
        info = inspect_archive(args.model)
        if args.predict:
            info["synthetic_score"] = check_prediction(args.model)
    except (OSError, ValueError, KeyError, IndexError, zipfile.BadZipFile,
            ImportError) as error:
        parser.exit(1, f"Smoke check failed: {error}\n")
    print(json.dumps(info, indent=2))
    print("PASS: archive checks" + (" and synthetic forward pass" if args.predict else ""))


if __name__ == "__main__":
    main()

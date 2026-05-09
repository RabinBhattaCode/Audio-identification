# Audio Identification

Shazam-style audio identification coursework for Music Informatics.

This project builds an audio fingerprinting system that indexes a database of recordings, fingerprints query snippets, and ranks likely matches using landmark hashes and offset voting.

## What It Does

- Loads and resamples audio with `librosa`.
- Builds log-magnitude STFT spectrograms.
- Detects local spectral peaks and constellation-map points.
- Converts anchor-target peak pairs into landmark hashes.
- Stores fingerprints in a pickle database.
- Matches query snippets against the fingerprint database.
- Evaluates retrieval with Top-1, Top-3, MAP@3, and rank counts.

## Tech Stack

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![librosa](https://img.shields.io/badge/librosa-111827?style=for-the-badge)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![SciPy](https://img.shields.io/badge/SciPy-8CAAE6?style=for-the-badge&logo=scipy&logoColor=white)
![scikit-image](https://img.shields.io/badge/scikit--image-f97316?style=for-the-badge)

## Main Files

```text
main.py                 Fingerprinting, database build, query matching, evaluation
eval.py                 Evaluation helper
requirements.txt        Python dependencies
fingerprints/           Saved fingerprint database
output.txt              Example output
```

## Install

```bash
pip install -r requirements.txt
```

## Notes

The implementation is based on Music Informatics lab material, constellation-map fingerprinting concepts, and Wang's landmark-hash audio search method.

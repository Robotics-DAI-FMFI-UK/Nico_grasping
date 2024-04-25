# SOM and MSOM Implementations

This repository contains Python implementations of a simple Self-Organizing Map (SOM) and Merge Self-Organizing Map (MSOM).

## Files

- `som.py`: Python implementation of a simple SOM.
- `msom.py`: Python implementation of a Merge SOM.

## Data

The `data` directory contains various data-related files:

- `dataset.json`: A JSON file containing all grasp data in a structured format. This file is created from text files in the `power`, `precision`, and `side` directories.
- `convert_text_file_to_json.py`: Python script for converting text files to JSON format.
- `histograms`: Python scripts for generating histograms.
- `power`: Text files containing power grasp data.
- `precision`: Text files containing precision grasp data.
- `side`: Text files containing side grasp data.

## Format of dataset.json

The `dataset.json` file follows a specific format:

```json
{
    "power": {
        "sequence 0": {
            "position 0": [],
            ...
            "position 15": []
        },
        ...
        "sequence 10": { ... }
    },
    "side": { ... },
    "precision": { ... }
}

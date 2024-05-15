# Master thesis repository

This repository contains datasets related to real and simulated grasps performed by the NICO robot. The datasets include:
- Datasets of Real and Simulated Robot NICO Grasps
- F5 and STSp grasps implemented by the NeuroNet library.
- UBAL implementation with input data.
- IK Solver program extended to compute coordinates from joint angles.
- Program for rotating 3D points.
- Data converters.
- IR sensor project: A program that enables a robot to perceive the surface of a table using an IR sensor.
- SOM and MSOM implemented in Python.




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

# Machine learning-based prediction of antiviral compounds for Chikungunya: Drug discovery using a combination of computational approaches.

## Overview
This repository contains a pipeline for predicting the antiviral activity of compounds using machine learning. The workflow involves generating molecular conformers, calculating molecular descriptors, and classifying compounds based on their antiviral potential.

## Workflow
1. **SMILES Input**: The user provides a SMILES string or a file containing multiple SMILES.
2. **SDF Generation**: The `sdf_convert.py` script converts SMILES into 3D molecular structures in SDF format.
3. **Descriptor Calculation**: The generated SDF file is processed using [PaDEL-Descriptor](http://yapcwsoft.com/dd/padeldescriptor/) to obtain molecular descriptors.
4. **Classification**: The output CSV from PaDEL-Descriptor is fed into `classifier.py`, which predicts whether the compound is "above_5" or "below_5" based on machine learning models.

## Installation
### Prerequisites
Ensure that the following dependencies are installed:
- Python 3.x
- RDKit
- NumPy
- Pandas
- tqdm
- scikit-learn

You can install them using:
```sh
pip install rdkit numpy pandas tqdm scikit-learn
```

### Clone the Repository
```sh
git clone https://github.com/Chikv-EC50-Classification.git
cd Chikv-EC50-Classification
```

## Usage
### Step 1: Generate SDF File
Run the following command to convert a SMILES string to an SDF file:
```sh
python sdf_convert.py -s "CCO" -o output.sdf
```
Or process multiple SMILES from a file:
```sh
python sdf_convert.py -f smiles_list.txt -o output.sdf
```

### Step 2: Compute Molecular Descriptors
- Download [PaDEL-Descriptor](http://yapcwsoft.com/dd/padeldescriptor/).
- When using the software to generate the descriptors use the following settings (disable the options that were not mentioned)
- - In the General tab
  - - Enable `1D & 2D`, `3D`, and `Fingerprints`
    - Enable `Remove salts` and `Standardize nitro groups`
- - In the `1D & 2D` and `3D` tab keep all the options enabled
  - In the `Fingerprints` tab only enable the `MACCSFingerprinter`
- Use it to generate a CSV file with molecular descriptors.

### Step 3: Classify the Compound
Run the classifier with the generated descriptor CSV:
```sh
python classifier.py -i descriptors.csv -o predictions.csv
```
The output file `predictions.csv` will contain the classification results.

## File Descriptions
- `sdf_convert.py`: Converts SMILES to SDF and optimizes molecular conformers.
- `classifier.py`: Loads trained models and classifies compounds based on molecular descriptors.
- `weights/models/`: Contains trained machine learning models.
- `weights/scalers/`: Contains the corresponding data scalers.

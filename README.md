# wekeo

[**Quickstart**](#Usage)
| [**Install guide**](#installation)

## Introduction

Wekeo Use Case Combined Processing Chain

## Preview

## Installation for Users

pip install ".[git]"

## Installation for Development

Made specifically for collaboration with Spascia.

### Environments

Several environment management methods are supported. Choose one of the following:

#### Virtual Environments (venv)

Create an empty virtual environment and activate it:

```bash
python -m venv .venv

# For macOS/Linux
source .venv/bin/activate
```

#### Conda Environments

Create a new conda environment (specify your Python version, e.g., `3.10`) and activate it:

```bash
conda create -n wekeo_env python=3.10 pip
conda activate wekeo_env
```

## Installation

Once your environment is set up and activated, in a clean folder, proceed in the following order

### 1. Clone the main repository

```bash
git clone https://github.com/hygeos/wekeo_combined_chain.git
```

### 2. Clone dependent repositories

Clone the specific data chain repositories into the project directory:

```bash
git clone https://github.com/hygeos/wekeo_frp_l3
git clone https://github.com/hygeos/wekeo_iasi_l3
git clone https://github.com/hygeos/wekeo_s5p_pca_l3
```

### 3. Install packages in editable mode

Install the dependencies so that changes to the code are reflected immediately:

```bash
pip install -e wekeo_combined_chain
pip install -e wekeo_frp_l3
pip install -e wekeo_iasi_l3
pip install -e wekeo_s5p_pca_l3
```

### Resulting folder structure

```
├── wekeo_combined_chain
├── wekeo_frp_l3
├── wekeo_iasi_l3
└── wekeo_s5p_pca_l3
```

Where each folder is installed as 'editable' in the python's environment, meaning that any changes inside will be reflected. 

## Prerequisites

### WEKEO HDA Credentials
You need to set up your WEKEO Harmonized Data Access (HDA) credentials in a file `~/.hdarc`

Required syntax:
```
user:your_username
password:your_password
```


### Environment Variables
Create a `.env` file in the project root directory with the following environment variables: DIR_ANCILLARY, OUTPUT_DIR

Example `.env` file:
```
DIR_ANCILLARY=/mnt/ceph/proj/WEKEO/ancillary
OUTPUT_DIR=/mnt/ceph/proj/WEKEO/outputs/
```

These variables define paths for storing downloaded data and generated outputs. The directories will be created automatically if they don't exist.

## Usage

The project includes a single Jupyter notebooks located in the `notebooks/` directory:

The chain caches results automatically and retrieve disk results if already presents. (can be disabled)

### combined.ipynb
This notebook performs the total wekeo use case processing chain
- s5p pca l3
- frp slstr l3 (+ download)
- iasi l3 (+ download)


can be run in different configurations depending on the parameters

    from wekeo_combined_chain.combined import get_combined_product

    ds = get_combined_product(...) # run the processing chain, cf notebook for more info

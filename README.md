# wekeo

[**Quickstart**](#Usage)
| [**Install guide**](#installation)

## Introduction

Wekeo Use Case Combined Processing Chain

## Preview

## Installation for Users

```bash
pip install ".[git]"
```

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

Install the chains in the python env as editable:

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
DIR_ANCILLARY=/data/WEKEO/ancillary
OUTPUT_DIR=/data/WEKEO/outputs/
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

### Cache

Most of the computations include a parameter **use_cache** that will bypass the computation if the result is already present (based on the parameters). 
So if you intend to modify the computations and test the result it is important to set **use_cache** to **False**. 

### FRP download issues

As of 02-04-2026 the download times for FRP have dramatically increased up to several hours for a single day.
This is an issue on the server, we have contacted the platform about it, they are investigating it. There is unfortunately not much more we can do about it.
But since the chains can also use download cache, you can copy it from someone who already downloaded the source files, and recompute the products yourself. 


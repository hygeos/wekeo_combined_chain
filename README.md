# wekeo

[**Quickstart**](#Usage)
| [**Install guide**](#installation)

## Introduction

Wekeo Use Case Combined Processing Chain

## Preview

---

## Installation

### Using pip
```bash
pip install -e .
```

### Using pixi
```bash
pixi install
```

### Prerequisites

#### WEKEO HDA Credentials
You need to set up your WEKEO Harmonized Data Access (HDA) credentials in a file `~/.hdarc`

Required syntax:
```
user:your_username
password:your_password
```


#### Environment Variables
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
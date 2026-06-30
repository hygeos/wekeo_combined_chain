# wekeo

[**Quickstart**](#Usage)
| [**Install guide**](#installation)

## Introduction

Wekeo Use Case Combined Processing Chain

## Preview

## Installation for Exploitation

### Clone the Repository
```bash
git clone https://github.com/hygeos/wekeo_combined_chain.git
cd wekeo_combined_chain
```

### Install Environment
Create the conda environment, install the project, and register the kernel:

```bash
# Create and activate the conda environment, then install the project
conda create -n wekeo && conda activate wekeo && pip install -e .
```

### Setup Environment Variables (`.env`)

Create a `.env` file at the root of the project and fill in your credentials:

```bash
touch .env
```

Add the following keys to the `.env` file (replace placeholders with your actual values):

```ini
OUTPUT_DIR=
AWS_ACCESS_KEY_ID=
AWS_SECRET_ACCESS_KEY=
AWS_DEFAULT_REGION=
```

## NOTE: below are instructions which matters only for production / development !

## Installation for Production

```bash
pip install ".[git]"
```

### Install coda specifically (important)

#### With conda

```
conda install coda
```

## Installation for Development

Made specifically for modifying the subchains while developping the combined chain

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

### Installation

Once your environment is set up and activated, in a clean folder, proceed in the following order

#### 1. Clone the main repository

```bash
git clone https://github.com/hygeos/wekeo_combined_chain.git
```

#### 2. Clone dependent repositories

Clone the specific data chain repositories into the project directory:

```bash
git clone https://github.com/hygeos/wekeo_frp_l3
git clone https://github.com/hygeos/wekeo_iasi_l3
git clone https://github.com/hygeos/wekeo_s5p_pca_l3
git clone https://github.com/hygeos/wekeo_plumes_post_process
```

#### 3. Install packages in editable mode

Install the chains in the python env as editable:

```bash
pip install -e wekeo_combined_chain
pip install -e wekeo_frp_l3
pip install -e wekeo_iasi_l3
pip install -e wekeo_s5p_pca_l3
pip install -e wekeo_plumes_post_process
```


#### 4. Install the coda library for iasi

##### With conda

```
conda install coda
```


#### Resulting folder structure

```
├── wekeo_combined_chain
├── wekeo_frp_l3
├── wekeo_iasi_l3
├── wekeo_s5p_pca_l3
└── wekeo_plumes_post_process

```

Where each folder is installed as 'editable' in the python's environment, meaning that any changes inside will be reflected. 

## Production / Development Notes:

### Requirements

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
DIR_ANCILLARY=/data/WEKEO/ancillary
OUTPUT_DIR=/data/WEKEO/outputs/

DIR_DOWNLOAD_COMBINED=/data/WEKEO/downloaded/     (used for the demo.ipynb notebook as a proxy for the S3 interfacing to come)
```

These variables define paths for storing downloaded data and generated outputs. The directories will be created automatically if they don't exist.

### Assessment

#### Cache

Most of the computations include a parameter **use_cache** that will bypass the computation if the result is already present (based on the parameters). 
So if you intend to modify the computations and test the result it is important to set **use_cache** to **False**. 

#### FRP download issues

As of 02-04-2026 the download times for FRP have dramatically increased up to several hours for a single day.
This is an issue on the server, we have contacted the platform about it, they are investigating it. There is unfortunately not much more we can do about it.
But since the chains can also use download cache, you can copy it from someone who already downloaded the source files, and recompute the products yourself. 


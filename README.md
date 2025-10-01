<div align="center">

# Downscaling Extreme Precipitation with Wasserstein Regularized Diffusion
<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>
[![Preprint](http://img.shields.io/badge/Preprint-arxiv.2410.00381-B31B1B.svg)](https://arxiv.org/abs/2410.00381)
[![Conference](http://img.shields.io/badge/Journal_Paper-IEEE_TGRS_(2025)-4b44ce.svg)](https://ieeexplore.ieee.org/document/11172297)
[![Google Colab](https://img.shields.io/badge/Colab%20Demo-F9AB00?logo=googlecolab&logoColor=fff)](https://colab.research.google.com/drive/1A07CzPGE1imSnkaQtU34fNIVIYMRXIWV?usp=sharing)

</div>

## Dependencies
To create a conda environment with all the required packages, run:
```
conda env create -f environment.yml
```

## Dataset Compilation
Instructions on how to obtain required training and validation data: 
### CPC Unified Gauge-Based Analysis of Daily Precipitation
Navigate to https://psl.noaa.gov/data/gridded/data.cpc.globalprecip.html, download `.nc` under precipitation.
Choose appropriate years.
The gauge density files are stored separately on NOAA's FTP server: https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/.
Download `.gz` files from https://ftp.cpc.ncep.noaa.gov/precip/CPC_UNI_PRCP/GAUGE_CONUS/RT/

### ERA5 and MRMS
Instructions can be found on this [repository](https://github.com/dossgollin-lab/climate-data).

### Prepare datasets
Once all the data are downloaded, navigate to `configs/local/default.yaml` and update these entries according to 
where the data are stored on your local machine: 
```yaml
# @package _global_
# This file is not tracked by git and is specific to YOUR MACHINE
# note: do not append / at the end of the path
local:
  data_dir: # PATH TO ROOT DATA
  log_dir: # PATH TO WANDB LOG
  eval_set_root_dir: # PATH TO EVAL OUTPUT
  specified_eval_root_dir: # PATH TO MATPLOTLIB OUTPUT
  model_root_dir: # PATH TO MODEL WEIGHTS
```

## Training
To train the proposed model (WassDiff), run
```python
python src/train.py trainer=gpu model=wassdiff experiment=det_val_sampler
```

To train the baseline diffusion model (SBDM, to be trained without Wasserstein Distance Regularization), run
```python
python src/train.py trainer=gpu model=ablation_wdr experiment=det_val_sampler
```

## Evaluation

Model weights can be found at
[Google Drive](https://drive.google.com/drive/folders/1mVHRyGTJDVZ_iS_yV0jVxmQs3bOkEoyB?usp=share_link).

Quantitative evaluation can be done by running 
```python
python ./src/eval.py trainer=gpu model=wassdiff experiment=eval_val_set ckpt_path=PATH_TO_MODEL_WEIGHTS name=NAME_OF_DIRECTORY
```

Set ckpt_path to the path of the model weights you want to evaluate, and name to the name of the directory 
where the evaluation results will be stored.
Note that the evaluation results will be stored in `eval_set_root_dir` specified in `configs/local/default.yaml`.

----------

To generate sample (single or ensemble) on a specified region and date (requires input data to be downloaded), run

```python
python ./src/eval.py model=wassdiff experiment=specified_eval
 ckpt_path=PATH_TO_MODEL_WEIGHTS name=NAME_OF_DIRECTORY
```

Set ckpt_path to the path of the model weights you want to evaluate, and name to the name of the directory 
where the evaluation results will be stored.
Note that the evaluation results will be stored in `specified_eval_root_dir` specified in `configs/local/default.yaml`.

To adjust ensemble size, append `model.num_samples=ENSEMBLE_SIZE` to command above.
You may modify the `lon`, `lat`, and `date` parameters in `configs/experiment/specified_eval.yaml` 
to specify the region and date of interest.

----------
You can optioanlly use `tiled_diffusion` to generate larger images (such as to for the entire CONUS region).
To do so, use `model=wassdiff_tiled` in the command below:
```python
python ./src/eval.py model=wassdiff experiment=specified_eval model=wassdiff_tiled ckpt_path=PATH_TO_MODEL_WEIGHTS name=NAME_OF_DIRECTORY
```


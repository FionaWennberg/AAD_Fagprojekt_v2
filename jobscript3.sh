#!/bin/bash
# ## General options 
# ## -- specify queue -- 
#BSUB -q hpc
# ## -- set the job Name -- 
#BSUB -J GAMMA
# ## -- ask for number of cores (default: 1) -- 
#BSUB -n 4 
# ## -- specify that the cores must be on the same host -- 
#BSUB -R "span[hosts=1]"
# ## -- specify that we need 4GB of memory per core/slot -- 
#BSUB -R "rusage[mem=8GB]"
# ## -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
#BSUB -M 8GB
# ## -- set walltime limit: hh:mm -- 
#BSUB -W 24:00 
# ## -- set the email address -- 
#BSUB -u s246024@dtu.dk
# ## -- send notification at start -- 
#BSUB -B 
# ## -- send notification at completion -- 
#BSUB -N 
# ## -- Specify the output and error file. %J is the job-id -- 
#BSUB -o Output_%J.out 
#BSUB -e Output_%J.err 

cd /work3/s246024/AAD_Fagprojekt_v2

source /work3/s246024/venv/bin/activate

export PYTHONPATH="$PWD:$PYTHONPATH"
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

python scripts/run_all_gamma_pipeline_updated.py \
  --stimdir /work3/jhjort/ds-eeg-snhl/stimuli \
  --base-processed-dir data/processed/env_onset \
  --spectrogram-dir data/processed/gamma/features \
  --processed-dir data/processed/gamma \
  --results-dir results_gamma \
  --subjects 31 32 33 34 35 36 37 38 39 40 41 42 43 \
  --step all \
  --max-workers 4 \
  --overwrite
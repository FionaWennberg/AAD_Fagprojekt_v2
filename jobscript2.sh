#!/bin/bash
# ## General options 
# ## -- specify queue -- 
#BSUB -q hpc
# ## -- set the job Name -- 
#BSUB -J ENV_ONSET
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
# module load python/3.11.9
source ../venv/bin/activate

export PYTHONPATH=/work3/s246024/AAD_Fagprojekt_v2:$PYTHONPATH

for SUBJECT in 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44
do
  python src/aad_project/preprocess_mtrf_envelope_onsets.py \
    --bidsdir /work3/jhjort/ds-eeg-snhl \
    --subject $SUBJECT \
    --out data/processed/env_onset/sub-$(printf "%03d" $SUBJECT)_mtrf_env_onset.npz
done

python scripts/run_all_env_onset.py \
  --processed-dir data/processed/env_onset \
  --results-dir results_env_onset \
  --subjects 1 2 3 4 5 6 7 8 9 10 11 12 13 14 15 16 17 18 19 20 21 22 23 24 25 26 27 28 29 30 31 32 33 34 35 36 37 38 39 40 41 42 43 44 \
  --max-workers 4 \
  --score-mode mean \
  --error l2 \
  --overwrite
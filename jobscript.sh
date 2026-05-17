!/bin/sh 
## General options 
## -- specify queue -- 
BSUB -q hpc
## -- set the job Name -- 
BSUB -J TRF_NH_HI
## -- ask for number of cores (default: 1) -- 
BSUB -n 4 
## -- specify that the cores must be on the same host -- 
BSUB -R "span[hosts=1]"
## -- specify that we need 4GB of memory per core/slot -- 
BSUB -R "rusage[mem=8GB]"
## -- specify that we want the job to get killed if it exceeds 5 GB per core/slot -- 
BSUB -M 8GB
## -- set walltime limit: hh:mm -- 
BSUB -W 24:00 
## -- set the email address -- 
please uncomment the following line and put in your e-mail address,
if you want to receive e-mail notifications on a non-default address
#BSUB -u s246024@dtu.dk
## -- send notification at start -- 
BSUB -B 
## -- send notification at completion -- 
BSUB -N 
## -- Specify the output and error file. %J is the job-id -- 
## -- -o and -e mean append, -oo and -eo mean overwrite -- 
BSUB -o Output_%J.out 
BSUB -e Output_%J.err 

cd /work3/s223643/AAD_FAGPROJEKT_V2
module load python/3.10
source venv/bin/activate

python scripts/run_all_subjects.py \
  --bidsdir /work3/jhjort/ds-eeg-snhl \
  --processed-dir data/processed \
  --results-dir results_backward \
  --step both \
  --max-workers 4
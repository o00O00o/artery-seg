cd ..
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=arterney-seg --kill-on-bad-exit=1 -w SH-IDC1-10-5-30-228\ 
python -u coarse.py\ 
--experiment_name coarse-all_label-baseline\ 
--all_label --baseline

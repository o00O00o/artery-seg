cd ..
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=arterney-seg --kill-on-bad-exit=1 -w SH-IDC1-10-5-30-222  python -u main.py --train_num 0.02 --val_num 0.78 --baseline 

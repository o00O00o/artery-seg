cd ..
srun -p MIA -n1 --gres=gpu:1 --mpi=pmi2 --job-name=arterney-seg --kill-on-bad-exit=1 \
-w SH-IDC1-10-5-30-228 \
python -u main.py \
--experiment_name fine-all_label-baseline-log_loss \
--stage fine --baseline --loss_func log_loss --all_label

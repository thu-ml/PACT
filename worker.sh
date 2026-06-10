export PYTHONPATH=$(pwd)
args_path=${1}
task_idx=${2}

PYTHONWARNINGS=ignore::UserWarning \
python -u rollout_worker.py --args_path ${args_path} --task_idx ${task_idx}
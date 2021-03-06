task=dphe_med
dir="/home/ch231037/r/DeepPhe_CR/med_system/medications/"
epochs=10
lr=2e-5
ev_per_ep=2
stl=0
seed=42
gas=4
encoder=roberta-base
dmy=$(date +'%m_%d_%Y')
logging_dir="/home/$USER/pipeline_models/logs/$task/$dmy/$encoder_name"
mkdir -p $logging_dir
logging_file="$logging_dir/ep_$epochs lr_$lr stl_$stl seed_$seed gas_$gas"
touch $logging_file
temp="/home/ch231037/pipeline_models/$task"
cache="/home/ch231037/pipeline_models/caches/$task"
mkdir -p $cache
mkdir -p $temp
python -m cnlpt.train_system \
--task_name $task \
--data_dir $dir \
--encoder_name $encoder \
--do_train \
--cache $cache \
--output_dir $temp \
--overwrite_output_dir \
--evals_per_epoch $ev_per_ep \
--do_eval \
--num_train_epochs $epochs \
--learning_rate $lr \
--seed $seed \
--gradient_accumulation_steps $gas >>$logging_file 2>&1

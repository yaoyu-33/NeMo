params=()
if [[ $MAX_STEPS -le 100 ]]; then # If greater than hundred we use defaults set in the training config file.
  LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
  VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
  LIMIT_VAL_BATCHES=`expr $MAX_STEPS / 20`
  params+=(training.trainer.log_every_n_steps=$LOG_EVERY_N_STEPS)
  params+=(training.trainer.limit_val_batches=$LIMIT_VAL_BATCHES)
  params+=(training.trainer.val_check_interval=$VAL_CHECK_INTERVAL)
fi
if [[ ! -z $LOCAL_NEMO_PATH ]]; then
  params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
fi
DATA_DIR=/lustre/fsw/joc/big_nlp/mt5/dataset/ci_data
PP_SPLIT_RANK=${PP_SPLIT_RANK:-`expr ${PP_SIZE} / 2`}

#TODO : Can add additional parameters (key value pairs from gitlab-ci.yaml file)
HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    training=${RUN_MODEL}/${RUN_MODEL_SIZE} \
    stages=["training"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${DATA_DIR} \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    training.run.name=${RUN_NAME} \
    training.run.time_limit=${TIME_LIMIT} \
    training.trainer.num_nodes=${NUM_NODES} \
    training.trainer.max_steps=${MAX_STEPS} \
    training.model.tensor_model_parallel_size=${TP_SIZE} \
    training.model.pipeline_model_parallel_size=${PP_SIZE} \
    training.model.pipeline_model_parallel_split_rank=${PP_SPLIT_RANK} \
    "${params[@]}" ${ADDITIONAL_PARAMS}
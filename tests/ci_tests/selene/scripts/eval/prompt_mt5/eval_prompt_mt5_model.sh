params=()
if [[ ! -z $LOCAL_NEMO_PATH ]]; then
  params+=("container_mounts=[${LOCAL_NEMO_PATH}:/opt/bignlp/NeMo]")
fi
PP_SPLIT_RANK=${PP_SPLIT_RANK:-`expr ${PP_SIZE} / 2`}

HYDRA_FULL_ERROR=1 BIGNLP_CI=1 python3 main.py \
    evaluation=${RUN_MODEL}/squad \
    stages=["evaluation"] \
    bignlp_path=${GIT_CLONE_PATH} \
    data_dir=${BASE_RESULTS_DIR}/data \
    base_results_dir=${BASE_RESULTS_DIR} \
    "container='${BUILD_IMAGE_NAME_SRUN}'" \
    cluster.partition=${SLURM_PARTITION} \
    cluster.account=${SLURM_ACCOUNT} \
    cluster.gpus_per_task=null \
    cluster.gpus_per_node=null \
    cluster.job_name_prefix="${SLURM_ACCOUNT}-bignlp_ci:" \
    evaluation.run.name=${RUN_NAME} \
    evaluation.run.time_limit=${TIME_LIMIT} \
    evaluation.run.model_train_name=${RUN_NAME} \
    evaluation.run.results_dir=${BASE_RESULTS_DIR}/${RUN_NAME} \
    evaluation.virtual_prompt_model_file=${BASE_RESULTS_DIR}/${PROMPT_LEARN_MODEL_DIR}/results/megatron_mt5_prompt.nemo \
    evaluation.language_model_path=${BASE_RESULTS_DIR}/${CONVERT_MODEL_DIR}/results/megatron_mt5.nemo \
    evaluation.tensor_model_parallel_size=${TP_SIZE} \
    evaluation.pipeline_model_parallel_size=${PP_SIZE} \
    evaluation.pipeline_model_parallel_split_rank=${PP_SPLIT_RANK} \
    "${params[@]}" ${ADDITIONAL_PARAMS}
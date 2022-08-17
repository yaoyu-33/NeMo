set -o xtrace

# Default values
MAX_STEPS=100
DATA_DIR=/lustre/fsw/joc/big_nlp/gpt3/prepare_dataset/the_pile/train
PRECISION=${PRECISION:-bf16}
CREATE_CHECKPOINT_CALLBACK_FLAG=False

case $RUN_MODEL_SIZE in

  126m)
    NUM_NODES=${NUM_NODES:-8}
    TP_SIZE=${TP_SIZE:-1}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  5b)
    NUM_NODES=${NUM_NODES:-20}
    TP_SIZE=${TP_SIZE:-2}
    PP_SIZE=${PP_SIZE:-1}
    ;;

  20b)
    NUM_NODES=${NUM_NODES:-80}
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-4}
    ;;

  40b)
    NUM_NODES=${NUM_NODES:-80}
    TP_SIZE=${TP_SIZE:-4}
    PP_SIZE=${PP_SIZE:-4}
    ;;

  175b)
    NUM_NODES=${NUM_NODES:-128}
    TP_SIZE=${TP_SIZE:-8}
    PP_SIZE=${PP_SIZE:-8}
    MAX_STEPS=50
    ;;

  *)
    echo -n "unknown"
    ;;
esac

LOG_EVERY_N_STEPS=`expr $MAX_STEPS / 100`
VAL_CHECK_INTERVAL=`expr $MAX_STEPS / 5`
LIMIT_VAL_BATCHES=`expr $MAX_STEPS / 20`

if [[ $AMP_STYLE = O1 ]]; then
  AMP_O2_FLAG=False
  GRADIENT_ACCUMULATION_FUSION_FLAG=False
else
  AMP_O2_FLAG=True
  GRADIENT_ACCUMULATION_FUSION_FLAG=True
  AMP_STYLE=O2 # by defualt all jobs use O2
fi

export RUN_NAME=${RUN_MODEL}_${RUN_MODEL_SIZE}_tp${TP_SIZE}_pp${PP_SIZE}_${NUM_NODES}nodes_${PRECISION}_precision_${AMP_STYLE}_${MAX_STEPS}steps
export RESULTS_DIR=${BASE_RESULTS_DIR}/${RUN_NAME}

HYDRA_FULL_ERROR=1 python3 main.py \
    +ci_test=True \
    training=${RUN_MODEL}/${RUN_MODEL_SIZE} \
    run_data_preparation=False \
    run_training=True \
    run_conversion=False \
    run_finetuning=False \
    run_evaluation=False \
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
    training.trainer.log_every_n_steps=${LOG_EVERY_N_STEPS} \
    training.trainer.val_check_interval=${VAL_CHECK_INTERVAL} \
    training.trainer.limit_val_batches=${LIMIT_VAL_BATCHES} \
    training.trainer.precision=${PRECISION} \
    training.model.tensor_model_parallel_size=${TP_SIZE} \
    training.model.pipeline_model_parallel_size=${PP_SIZE} \
    training.model.megatron_amp_O2=${AMP_O2_FLAG} \
    training.model.gradient_accumulation_fusion=${GRADIENT_ACCUMULATION_FUSION_FLAG} \
    training.exp_manager.create_checkpoint_callback=${CREATE_CHECKPOINT_CALLBACK_FLAG}
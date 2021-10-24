#!/bin/bash

ERROR_FILE="error_per_class.txt"
TMP_DIR=$4 # for temporary log file
mkdir -p ${TMP_DIR}
python duplex_text_normalization_test.py data.test_ds.data_path=$3 decoder_pretrained_model=$1 tagger_pretrained_model=$2 data.test_ds.use_cache=True data.test_ds.batch_size=32 lang=en mode=tn #2>&1 | tee ${TMP_DIR}/log_errors.txt
mv ${ERROR_FILE} ${TMP_DIR}/.
declare -a cls=("CARDINAL" "DECIMAL" "ORDINAL" "DIGIT" "TIME" "DATE" "ADDRESS" "TELEPHONE" "MEASURE" "MONEY" "FRACTION")
for c in "${cls[@]}"
do
    echo "finding unrecoverable errors for ${c}"
    grep "^${c}\s" ${TMP_DIR}/${ERROR_FILE} > ${TMP_DIR}/log_errors_${c}.txt
    python find_unrecoverable_errors.py ${TMP_DIR}/log_errors_${c}.txt
done


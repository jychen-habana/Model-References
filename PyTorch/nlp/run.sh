#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`
WORK_SPACE=`cd ${BIN};cd ../../; pwd`

LLM_BS="1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128"
INPUT_LENGTH="32 128 256 512 1024 1536 1792 1920"
MAX_LENGTH="512 1024 2048"

TMP_DIR=${BIN}/tmp
rm -rf ${TMP_DIR}
mkdir -p ${TMP_DIR}

cd ${TMP_DIR}

git clone https://github.com/jychen-habana/Model-References.git
MODEL_REFERENCE_DIR=${TMP_DIR}/Model-References
cd ${MODEL_REFERENCE_DIR}
git checkout perf_measure_ChatGLM2-6B

if [ "${CHATGLM_TYPE}" == "ChatGLM-6B" ]
then
    echo "==================Running ChatGLM-6B=================="

    CHATGLM_6B_DIR=${MODEL_REFERENCE_DIR}/PyTorch/nlp/ChatGLM-6B
    cd ${CHATGLM_6B_DIR}
    python -m pip install -r ${CHATGLM_6B_DIR}/requirements.txt

    CMD="python ${CHATGLM_6B_DIR}/cli_demo.py --prompt ${RUN_PROMPT} --max_new_tokens ${MAX_NEW_TOKENS} --repeat ${RUN_REPEAT} --batch_size ${RUN_BATCH_SIZE}"
    if [ "${OFFLINE_MODE}" == false ]
    then
        ${CMD}
    fi

    cp -rf ${CHATGLM_6B_DIR}/models--THUDM--chatglm-6b/* ${TRANSFORMERS_CACHE}/models--THUDM--chatglm-6b/snapshots/*/
    rm -rf ${HOME}/.cache/huggingface/modules/transformers_modules/THUDM
    ${CMD}

fi

if [ "${CHATGLM_TYPE}" == "ChatGLM2-6B" ]
then
    echo "==================Running ChatGLM2-6B=================="

    CHATGLM2_6B_DIR=${MODEL_REFERENCE_DIR}/PyTorch/nlp/ChatGLM2-6B
    cd ${CHATGLM2_6B_DIR}
    python -m pip install -r ${CHATGLM2_6B_DIR}/requirements.txt

    CMD="python ${CHATGLM2_6B_DIR}/cli_demo.py --prompt ${RUN_PROMPT} --max_new_tokens ${MAX_NEW_TOKENS} --repeat ${RUN_REPEAT} --batch_size ${RUN_BATCH_SIZE}"
    if [ "${OFFLINE_MODE}" == false ]
    then
        ${CMD}
    fi

    cp -rf ${CHATGLM2_6B_DIR}/models--THUDM--chatglm2-6b/* ${TRANSFORMERS_CACHE}/models--THUDM--chatglm2-6b/snapshots/*/
    rm -rf ${HOME}/.cache/huggingface/modules/transformers_modules/THUDM
    ${CMD}

fi

rm -rf ${TMP_DIR}


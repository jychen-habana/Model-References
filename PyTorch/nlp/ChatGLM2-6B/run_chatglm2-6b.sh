#!/bin/bash
set -ex

BIN=`dirname ${0}`
BIN=`cd ${BIN}; pwd`

# MAX_LENGTH="512 1024 2048"
# INPUT_LENGTH="32 128 256 512 1024 1536 1792 1920"
# LLM_BS="1 2 4 8 16 24 32 40 48 56 64 72 80 88 96 104 112 120 128"
RUN_REPEAT="5"
LLM_BS="1 2 4"
INPUT_LENGTH="32 128"
MAX_LENGTH="512 1024"

if [ "${OFFLINE_MODE}" == true ]
then
    export TRANSFORMERS_OFFLINE=1
else
    export -n TRANSFORMERS_OFFLINE
fi

TMP_DIR=${BIN}/tmp
rm -rf ${TMP_DIR}
mkdir -p ${TMP_DIR}

cd ${TMP_DIR}

git clone https://github.com/jychen-habana/Model-References.git
MODEL_REFERENCE_DIR=${TMP_DIR}/Model-References
cd ${MODEL_REFERENCE_DIR}
git checkout auto_multi_prompt_perf_measure

echo "==================Running ChatGLM2-6B=================="

CHATGLM2_6B_DIR=${MODEL_REFERENCE_DIR}/PyTorch/nlp/ChatGLM2-6B
cd ${CHATGLM2_6B_DIR}
python -m pip install -r ${CHATGLM2_6B_DIR}/requirements.txt

CMD="python ${CHATGLM2_6B_DIR}/cli_demo.py"
if [ "${OFFLINE_MODE}" == false ]
then
    ${CMD} || echo "Finished weights downloading!"
fi
cp -rf ${CHATGLM2_6B_DIR}/models--THUDM--chatglm2-6b/* ${HOME}/.cache/huggingface/hub/models--THUDM--chatglm2-6b/snapshots/*/
rm -rf ${HOME}/.cache/huggingface/modules/transformers_modules/THUDM/chatglm2-6b

for l in ${MAX_LENGTH}
do
    for i in ${INPUT_LENGTH}
    do
        for b in ${LLM_BS}
        do
            for r in ${RUN_REPEAT}
            do
                ${CMD} --max_length ${l} --prompt ${i} --batch_size ${b} --repeat ${r} 2>&1 | tee ${HOME}/chatglm2-6b_${l}_${i}_${b}_${r}.log
            done
        done
    done
done

rm -rf ${TMP_DIR}


#!/bin/sh

## MODIFY PATH for YOUR SETTING


DATA_ROOT=$1
LIST_FILE=$2
OUT_DIR=$3
DEV_ID=$4
MODEL=$5
ROOT_DIR=$6
CROP_SIZE=$7
pickyEye=$8

echo $pickyEye
echo $DATA_ROOT
echo $LIST_FILE
echo $OUT_DIR
if [ ! -d ${OUT_DIR} ]; then
    mkdir -p ${OUT_DIR};
    echo creat dir $OUT_DIR
fi

EXP=pickyEye
NET_ID=mod9
STORE_ID=whole1228
NUM_LABELS=2
HOLE0=2
HOLE1=4
HOLE2=6
HOLE3=8
RUN_TEST=1
MEAN0=123.9
MEAN1=123.9
MEAN2=123.9
CAFFE_DIR=$ROOT_DIR
CAFFE_BIN=${CAFFE_DIR}/build/tools/caffe


TRAIN_SET_SUFFIX=_aug
TEST_SET=val
TEST_ITER=`cat ${LIST_FILE} | wc -l`


sed "$(eval echo $(cat ${pickyEye}/sub.sed))" ${pickyEye}/test.prototxt > ${pickyEye}/test_${TEST_SET}.prototxt

CMD="${CAFFE_BIN} test \
             --model=${pickyEye}/test_${TEST_SET}.prototxt \
             --weights=${MODEL} \
             --gpu=${DEV_ID} \
             --iterations=${TEST_ITER}"
echo Running ${CMD} && ${CMD}



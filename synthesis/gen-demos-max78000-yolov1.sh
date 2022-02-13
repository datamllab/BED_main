#!/bin/sh
DEVICE="MAX78000"
CHECKPOINT="../../ai8x-training/yolov1/Yolov1_checkpoint-q.pth.tar"
TARGET="test_sdk"
COMMON_ARGS="--device $DEVICE --compact-data --mexpress --timer 0 --display-checkpoint"

./ai8xize.py --verbose --log --overwrite --fifo --test-dir $TARGET --prefix yolov1 --checkpoint-file $CHECKPOINT --config-file networks/yolo-224-hwc-ai85_MXIM.yaml $COMMON_ARGS "$@"

# add --prefix energy to measure power consumption, or it will measure inference time

# add --softmax to enable softmax for the last layer

# --fifo


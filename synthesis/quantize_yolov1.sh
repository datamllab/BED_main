#!/bin/sh
./quantize.py ../../ai8x-training/yolov1/Yolov1_checkpoint.pth.tar ../../ai8x-training/yolov1/Yolov1_checkpoint-q.pth.tar --device MAX78000 -v -c networks/yolo-224-hwc-ai85_MXIM.yaml "$@"

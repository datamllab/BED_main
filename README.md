## BED: A Real-Time Object Detection System for Edge Devices

### Prerequirement

Before, it is necessary to install the envoironment of [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training) and [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis) in different branches.

### Train a tiny model

````angular2html
conda activate ai8x-training
python train/YOLO_V1_Train_QAT.py
````

### Synthesis

````angular2html
conda activate ai8x-synthesis
python train/YOLO_V1_Train_QAT.py
````

### Deploy the model to MAX78000 using BED GUI



### Deploy the model to MAX78000 for real-time object detection



### Evaluation and Demonstration

#### Offline Evaluation

<div align=center>
<img width="400" height="270" src="https://github.com/datamllab/BED_main/blob/main/figure/offline_results2.png">
</div>

#### Real-time demonstration

For real-time demonstration, please go to see our demo video.


### Acknowledgement


### Cite this work

If you find this repo useful, you can cite by:

````angular2html

````
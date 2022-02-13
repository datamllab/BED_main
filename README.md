## BED: A Real-Time Object Detection System for Edge Devices
<img width="400" height="200" src="https://github.com/datamllab/BED_main/blob/main/figure/BED_logo.png">


### About this project

This project focus on end-to-end oBject
detection system for Edge Devices (BED).
BED integrates a deep nerual network (DNN) practiced on [MAX78000](https://www.maximintegrated.com/en/products/microcontrollers/MAX78000.html) with I/O devices, as illustrated in the following figure. 
The DNN model for the detection is deployed on MAX78000; 
and the I/O devices includes a camera and a screen for image acquisition and output exhibition, respectively. 

<div align=center>
<img width="600" height="200" src="https://github.com/datamllab/BED_main/blob/main/figure/sys_config-p.png">
</div>

### Train a tiny model

Before training the model, it is necessary to clone and install the envoironment of [ai8x-training](https://github.com/MaximIntegratedAI/ai8x-training).
Once you finished the installation, you can use this commend to train a model:

````angular2html
conda activate ai8x-training
python train/YOLO_V1_Train_QAT.py
````

### Synthesis

Before the synthesis of pretrained model, it is necessary to clone and install the environment of [ai8x-synthesis](https://github.com/MaximIntegratedAI/ai8x-synthesis) in a different branch.

Once you finished the installation, it is required to add the following files to the local directory of ai8x-synthesis: 

* Put [sample_yolov1.npy](https://github.com/datamllab/BED_main/blob/main/synthesis/sample_yolov1.npy) into the directory ./test/ of ai8x-synthesis. 
* Put [quantize_yolov1.sh](https://github.com/datamllab/BED_main/blob/main/synthesis/quantize_yolov1.sh) into the directory ./scripts/ of ai8x-synthesis.
* Put [yolo-224-hwc-ai85_MXIM.yaml](https://github.com/datamllab/BED_main/blob/main/synthesis/yolo-224-hwc-ai85_MXIM.yaml) into the directory ./networks/ of ai8x-synthesis.
* Put [gen-demos-max78000-yolov1.sh](https://github.com/datamllab/BED_main/blob/main/synthesis/gen-demos-max78000-yolov1.sh) into the directory ./ of ai8x-synthesis

With all above steps finished, you can use this commend to quantize the pretrained model: 
````angular2html
conda activate ai8x-synthesis
sh ./scripts/quantize_yolov1.sh
````

After the quantization, you can use this commend to synthesize the pretrained model: 
````angular2html
sh gen-demos-max78000-yolov1.sh
````

### Deploy the model to MAX78000 using BED GUI

Please follow the [tutorial](https://github.com/datamllab/BED_GUI) to use BED GUI to deploy the model to the MAX78000. 

### Deploy the model to MAX78000 for real-time object detection

Please follow the [tutorial](https://github.com/datamllab/BED_camera) to use the MAX78000 for real-time object detection.

### Evaluation and Demonstration

#### Offline Evaluation

We focus on the case study for the offline evaluation. The detection results for the randomly selected images from the [VOC2007](http://host.robots.ox.ac.uk/pascal/VOC/voc2007/index.html) testing dataset are given as follows: 

<div align=center>
<img width="1000" height="180" src="https://github.com/datamllab/BED_main/blob/main/figure/offline_results2.png">
</div>

#### Real-time demonstration

BED shows the real-time detection results on the screen of the board. Here, we select several results as follows:

<div align=center>
<img width="1000" height="200" src="https://github.com/datamllab/BED_main/blob/main/figure/real_results_more.png">
</div>

For detailed demonstration, please go to see our [demo video]().


### Acknowledgement

TBD.

### Cite this work

If you find this repo useful, you can cite by:

````angular2html
TBD.
````
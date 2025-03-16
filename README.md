# Using mmdection for traffic scense detectoion (Graduation Thesis)

## MMDetection Commands

## Verify Installed Versions and CUDA Availability

Run the following commands to check your installed versions:

```python
import torch
import mmcv

print("Python Version:", "3.8.20")
print("Torch Version:", torch.__version__)       # Should print 2.0.1+cu118
print("CUDA Version:", torch.version.cuda)      # Should print 11.8
print("MMCV Version:", mmcv.__version__)        # Should print 2.0.1
print("CUDA Available:", torch.cuda.is_available())  # Should print True
```

## Running Image Inference

### RTMDet Model
To perform inference using the RTMDet model on an image with CPU:

```sh
python demo/image_demo.py demo/image.png rtmdet_tiny_8xb32-300e_coco.py \
--weights rtmdet_tiny_8xb32-300e_coco_20220902_112414-78e30dcc.pth --device cpu
```

### ViTDet Model
To perform inference using the ViTDet model with CUDA:

```sh
python demo/image_demo.py demo/demo.jpg projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
--weights vitdet_mask-rcnn_vit-b-mae_lsj-100e_20230328_153519-e15fe294.pth --device cuda
```

To infer an image using a fine-tuned ViTDet model:

```sh
python demo/image_demo.py demo/image-small-vit.png projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
--weights work_dirs/vitdet_mask-rcnn_vit-b-mae_lsj-100e-small/iter_500.pth --device cuda
```

## Training with a Custom Dataset using ViTDet Model

### Steps to Prepare the Dataset
1. Create a directory named `data` under the `mmdetection` directory.
2. Inside `data`, place the COCO dataset structure:
   - `train2017/`
   - `val2017/`
   - `annotations/`
3. Download the `mae_pretrain_vit_base.pth` file and place the this checkpoint in `vitdet_mask-rcnn_vit-b-mae_lsj-100e.py` in the configuration file.

### Train the Model
Use the following command to train the model with the custom dataset:

```sh
python tools/train.py projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py
```

## Running Inference with a Specific Checkpoint

To infer an image using a trained model at a specific iteration:

```sh
python demo/image_demo.py demo/image.png projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
--weights work_dirs/vitdet_mask-rcnn_vit-b-mae_lsj-100e/iter_10000.pth --device cuda
```


### Experiement Result
### original image
![demo1](demo/t2.png)
### demo of official model 
config:
```python
image_size = (1024, 1024)
train_loader: batch_size = 4, number of worker = 8
val_loader : batch_size =1 , numnber of worker = 4
186736 iteration
```
![Experiment 1 Result](output/t2-official.png)

### Experiment 1
config:
```python
image_size = (512, 512)
train_loader: batch_size = 1, number of worker = 2
val_loader : batch_size =1 , numnber of worker = 2
18736 iteration
```
![Experiment 2 Result](output/t2.png)
### Experiment 2
config:
```python
image_size = (1024, 1024)
train_loader: batch_size = 1, number of worker = 4
val_loader : batch_size =1 , numnber of worker = 2
10000 iteration
```
![Experiment 2 Result](output/t2-e2.png)

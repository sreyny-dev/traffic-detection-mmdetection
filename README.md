# traffic-detection-mmdetection

# MMDetection Commands

## Verify Installed Versions and CUDA Availability

Run the following commands to check your installed versions:

```python
import torch
import mmcv

print(torch.__version__)       # Should print 2.0.1+cu118
print(torch.version.cuda)      # Should print 11.8
print(mmcv.__version__)        # Should print 2.0.1
print(torch.cuda.is_available())  # Should print True
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

## Training the ViTDet Model

To start training using the ViTDet model configuration:

```sh
python tools/train.py projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py
```

## Running Inference with a Specific Checkpoint

To infer an image using a trained model at a specific iteration:

```sh
python demo/image_demo.py demo/image.png projects/ViTDet/configs/vitdet_mask-rcnn_vit-b-mae_lsj-100e.py \
--weights work_dirs/vitdet_mask-rcnn_vit-b-mae_lsj-100e/iter_18437.pth --device cuda

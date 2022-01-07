## CR-Aug

The demo code for our paper **AUGMENTATION-INDUCED CONSISTENCY REGULARIZATION FOR CLASSIFICATION**

## Requirement

```
conda create -n CR-Aug python=3.6
conda install pytorch=1.2 cudatoolkit=10.0 torchvision
pip install tensorboard==1.14.0
```

## Train

Pre-trained checkpoint of our model can be found from [google drive](https://drive.google.com/drive/folders/1JB3iDktfo3-ZjfHPM3jXRUtjDSuUYXgD?usp=sharing) and put it in the folder(-->pretrain_model)

1. baseline

   ```
   python baseline.py --epoch=40 --lr=0.01
   ```

2. CR-Aug

   ```
   python CR-Aug.py --transforms=combine --lr=0.001
   ```

   
### GNR638 Course Project
## Shubham Lohiya, Aditya Iyengar, Sharvaree Sinkar

```
├── README.md  
├── code  
│   ├── ResNet_backbone.py  
│   ├── SINet.py  
│   ├── dataloader.py  
│   ├── search_attention.py  
│   ├── test.py  
│   ├── train.py  
│   └── trainer.py  
└── notebooks  
    ├── camouflaged_object_detection_main.ipynb  
    └── camouflaged_object_detection_modified.ipynb  

- dataloader.py (converts training and test datasets into iterable *dataloader* objects  
- ResNet_backbone.py (structure of the implemented ResNet for detection)  
- search_attention.py (architecture of the search attention module)  
- SINet.py (complete architecture of the model in the paper)  
- test.py (testing the model on the loaded test set)  
- train.py (training the model on the loaded train set)  
- trainer.py (defines all training parameters)  

usage: MyTest.py [-h] [--testsize TESTSIZE] [--model_path MODEL_PATH]
                 [--test_save TEST_SAVE]

optional arguments:
  -h, --help            show this help message and exit  
  --testsize TESTSIZE   the snapshot input size  
  --model_path MODEL_PATH
                        path of saved model  
  --test_save TEST_SAVE
                        location to save output data  


usage: MyTrain.py [-h] [--epoch EPOCH] [--lr LR] [--batchsize BATCHSIZE]
                  [--trainsize TRAINSIZE] [--clip CLIP]
                  [--decay_rate DECAY_RATE] [--decay_epoch DECAY_EPOCH]
                  [--gpu GPU] [--save_epoch SAVE_EPOCH]
                  [--save_model SAVE_MODEL] [--train_img_dir TRAIN_IMG_DIR]
                  [--train_gt_dir TRAIN_GT_DIR]

optional arguments:
  -h, --help            show this help message and exit  
  --epoch EPOCH         epoch number, default=30  
  --lr LR               init learning rate, try `lr=1e-4`  
  --batchsize BATCHSIZE
                        training batch size (Note: ~500MB per img in GPU)  
  --trainsize TRAINSIZE
                        the size of training image, try small resolutions for
                        speed (like 256)  
  --clip CLIP           gradient clipping margin  
  --decay_rate DECAY_RATE
                        decay rate of learning rate per decay step  
  --decay_epoch DECAY_EPOCH
                        every N epochs decay lr  
  --gpu GPU             choose which gpu you use  
  --save_epoch SAVE_EPOCH
                        every N epochs save your trained snapshot  
  --save_model SAVE_MODEL
                        location where to save model  
  --train_img_dir TRAIN_IMG_DIR
                        location of training data  
  --train_gt_dir TRAIN_GT_DIR
                        location of training ground truths  
```

### Submission link and presentation video can be found [here](https://drive.google.com/drive/folders/1V9DqR_Ve9XxTKO77iSObGlJFpTghwx3R).

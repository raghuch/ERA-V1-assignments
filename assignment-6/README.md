## Solution to Assignment-6

This dir contains the solution to assignment-6 in ERA-v1 course. Steps to obtain 99.4% accuracy on MNIST test set within 20k params model size, and in 20 epochs.

### Dataset

Dataset (MNIST) transforms: For training set, normalize, convert to Tensor and random rotate upto 7 degrees. For test set, just normalize and convert to tensor. Used the torchvision transforms library.

### Model architecture

Model: Simple CNN with 4 blocks.
 * initial input block: conv2d (1 channel -> 16 channels, kernel=3, padding=0) -> relu -> batchnorm -> dropout(0.15)
 * conv block 1: conv2d (16 channels -> 32 channels, kernel=3, padding=0) -> relu -> batchnorm -> dropout(0.15)
 * 'transitin block': conv2d(32 channels -> 20 channels, kernel=1, padding=0) -> maxpool(2,2) # features are 'mixed'  and then size is halved
 * conv block 2: conv2d (20 channels -> 32 channels, kernel=3, padding=0) -> relu -> batchnorm -> dropout(0.15)
 * conv block 3: conv2d (32 channels -> 16 channels, kernel=3, padding=0) -> relu -> batchnorm -> dropout(0.15)
 * conv block 4: conv2d (16 channels -> 16 channels, kernel=3, padding=0) -> relu -> batchnorm -> dropout(0.15) #output size is 6
 * output block which contains:
   * global avg pool block: Avgpool2d(kernel_size=(6,6)) # size reduced to 1 with 16 channels
   * conv2d(16 channels -> 10 channels, kernel=1, padding=0) #size 1 with 10 channels. will change shape to batch_sz x 10 later as view(-1, 10)

Final layer & loss : log_softmax with NLL loss.
Optimizer: SGD with lr = 0.01, momentum = 0.9. No lr scheduler was used.

### Model size

Model size: torchsummary prints shows 18.5k params as shown in [this notebook](https://github.com/raghuch/ERA-V1-assignments/blob/main/assignment-6/assignment6.ipynb)

----------------------------------------------------------------
        Layer (type)               Output Shape         Param #
----------------------------------------------------------------
            Conv2d-1           [-1, 16, 26, 26]             144
              ReLU-2           [-1, 16, 26, 26]               0
       BatchNorm2d-3           [-1, 16, 26, 26]              32
         Dropout2d-4           [-1, 16, 26, 26]               0
            Conv2d-5           [-1, 32, 24, 24]           4,608
              ReLU-6           [-1, 32, 24, 24]               0
       BatchNorm2d-7           [-1, 32, 24, 24]              64
         Dropout2d-8           [-1, 32, 24, 24]               0
            Conv2d-9           [-1, 20, 24, 24]             640
        MaxPool2d-10           [-1, 20, 12, 12]               0
           Conv2d-11           [-1, 32, 10, 10]           5,760
             ReLU-12           [-1, 32, 10, 10]               0
      BatchNorm2d-13           [-1, 32, 10, 10]              64
        Dropout2d-14           [-1, 32, 10, 10]               0
           Conv2d-15             [-1, 16, 8, 8]           4,608
             ReLU-16             [-1, 16, 8, 8]               0
      BatchNorm2d-17             [-1, 16, 8, 8]              32
        Dropout2d-18             [-1, 16, 8, 8]               0
           Conv2d-19             [-1, 16, 6, 6]           2,304
             ReLU-20             [-1, 16, 6, 6]               0
      BatchNorm2d-21             [-1, 16, 6, 6]              32
        Dropout2d-22             [-1, 16, 6, 6]               0
        AvgPool2d-23             [-1, 16, 1, 1]               0
           Conv2d-24             [-1, 10, 1, 1]             160
----------------------------------------------------------------

Total params: 18,448                                             
Trainable params: 18,448                                        
Non-trainable params: 0                                         

----------------------------------------------------------------

Input size (MB): 0.00                                           
Forward/backward pass size (MB): 1.15                           
Params size (MB): 0.07                                          
Estimated Total Size (MB): 1.22                                     

----------------------------------------------------------------


Train and Test loop: Exactly the same as in the reference notebook. The code is divided into the [model.py](https://github.com/raghuch/ERA-V1-assignments/blob/main/assignment-6/model.py) which contains the CNN and the [utils.py](https://github.com/raghuch/ERA-V1-assignments/blob/main/assignment-6/utils.py) with data loaders, train and test loops in the same dir.

The code is run in the [assignment6.ipynb](https://github.com/raghuch/ERA-V1-assignments/blob/main/assignment-6/assignment6.ipynb) file which has training logs. The maximum accuracy is at 19th epoch, printed in the notebook, also shown here:


### Maximum accuracy

loss=0.1573498398065567 batch_id=468: 100%|██████████| 469/469 [00:01<00:00, 303.77it/s]  

Test set: Average loss: 0.0215, Accuracy: 9933/10000 (99.33%)

Running EPOCH: 19

loss=0.01866978220641613 batch_id=468: 100%|██████████| 469/469 [00:01<00:00, 305.87it/s] 

Test set: Average loss: 0.0194, Accuracy: 9940/10000 (99.40%)

Running EPOCH: 20

loss=0.044775012880563736 batch_id=468: 100%|██████████| 469/469 [00:01<00:00, 306.18it/s]

Test set: Average loss: 0.0190, Accuracy: 9934/10000 (99.34%)


Probably we can obtain better stability in test loss by using an LR scheduler.

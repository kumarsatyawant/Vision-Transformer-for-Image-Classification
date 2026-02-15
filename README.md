# Vision Transformer (ViT) 

- We trained ViT network using CIFAR-10 dataset for Image classification. 

- We adopted two approaches for training on the CIFAR-10. First, we trained the network by randomly assigning its weights. Second, we used transfer learning to finetune the trained network on ImageNet daatset.


Network:
------- 
![Network](figures/5_ViT.png)  
Fig. 1. The schematic representation of the ViT network. The Image source is taken from [AN IMAGE IS WORTH 16X16 WORDS: TRANSFORMERS FOR IMAGE RECOGNITION AT SCALE](https://arxiv.org/abs/2010.11929).


Results:
-----------------------------------------

We achieved test accuracy of 70.97% and 80.97% when using random initialization and transfer learning respectively.  


![Results](figures/1_Train_Test_Accuracy_Random_Initialization.jpg) 
Fig. 2. Plot for Training and Testing Accuracies when uisng Random Weight Initialization.


![Results](figures/2_Train_Test_Loss_Random_Initialization.jpg) 
Fig. 3. Plot for Training and Testing Losses when uisng Random Weight Initialization.


![Results](figures/3_Train_Test_Accuracy_Transfer_Learning.jpg) 
Fig. 4. Plot for Training and Testing Accuracies when using Transfer Learning.


![Results](figures/4_Train_Test_Loss_Transfer_Learning.jpg) 
Fig. 5. Plot for Training and Testing Losses when using Transfer Learning.



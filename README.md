<!-- ![alt text](https://github.com/hyukahn16/adv_bilevel_optim/blob/master/saved_models/pgd_merge/pgd_accuracy.png) -->
| ResNet18 trained w/ PGD<sup>10</sup>  | ResNet18 traind w/ BETA<sup>10</sup> |
| ------------- | ------------- |
| <img src="https://github.com/hyukahn16/adv_bilevel_optim/blob/v1/saved_models/pgd_merge/pgd_accuracy.png" width="500" height="400"/>  | <img src="https://github.com/hyukahn16/adv_bilevel_optim/blob/v1/saved_models/bilevel_merge/bilevel_accuracy.png" width="500" height="400"/>  |

*CIFAR-10 Dataset  
*Both models tested with PGD<sup>20</sup>  
("Test Robust Accuracy" == PGD<sup>20</sup> accuracy)

*Learning rate: 0.05 from Epoch 0-100 | 0.005 from Epoch 100-150  
*Optimizer: RMSProp for BETA-trained model

Notes:  
BETA-trained model seems to overfit more to its training data than PGD-trained model.  
Potentially, it may be due to use of optimizer.

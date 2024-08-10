<!-- ![alt text](https://github.com/hyukahn16/adv_bilevel_optim/blob/master/saved_models/pgd_merge/pgd_accuracy.png) -->
| ResNet18 trained w/ PGD<sup>10</sup>  | ResNet18 traind w/ BETA<sup>10</sup> |
| ------------- | ------------- |
| <img src="https://github.com/hyukahn16/adv_bilevel_optim/blob/master/saved_models/pgd_merge/pgd_accuracy.png" width="500" height="400"/>  | <img src="https://github.com/hyukahn16/adv_bilevel_optim/blob/master/saved_models/bilevel_merge/bilevel_accuracy.png" width="500" height="400"/>  |

Experiment Info:
- CIFAR-10 Dataset (Train and Test)
- Both models trained with 10 attack iterations of their corresponding adversary
- Both models tested with PGD<sup>20</sup> ("Test Robust Accuracy" == PGD<sup>20</sup> accuracy)
- Learning rate: 0.1 from Epoch 0-100 | 0.001 from Epoch 100-150
- Optimizer: RMSProp for BETA-trained model

Notes:  
- BETA-trained model shows better robustness to PGD<sup>20</sup> than PGD<sup>10</sup>-trained model  
- Need to find more optimal learning rate for BETA model

---
<img src="https://github.com/hyukahn16/adv_bilevel_optim/blob/master/saved_models/beta_001/beta_accuracy.png" width="500" height="400"/>

Experiment Info:
- ResNet learning rate: 0.001 from Epoch 0-100 | 0.0001 from Epoch 100-150
- BETA learning rate: 2/255

Notes:
- Low accuracy compared to other learning rates
- Stuck in local minima?
- No overfitting?

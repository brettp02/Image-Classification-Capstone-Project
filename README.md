# Computer Vision - Image Classification Capstone Project
* * *
This repository contains all code for my Image Classification capstone project:
- `code` - contains `train.py`, `test.py`, `testdata` folder and `model.pth` - a saved file containing the best model from training. Inside of train.p you can select between my custom implementation, resnet50 trained with transfer learning, or an ensemble of both of these. *Note: if you would like to train the model/s on different training datasets where there aren't 3 classes you will have to modify the output classes for resnet and for NeuralNet.*
- `jupyter notebooks` - contains the notebooks I used for performing EDA and data cleaning.
- `datasets` - contains the processed data, which consists of images of tomatos, strawberries, and cherrys. If you would like to test the model on your own images, you must follow this same structure inside of `./code/testdata`.
- `report.pdf` - contains all required information on the project including model architecture, preprocessing steps, and performance metrics.

### To run test.py or train.py:
- `python3 train.py`
- `python3 test.py`

*The expected test accuracy on the current training dataset is ~90%*

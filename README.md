# HTR_On_Railway_Forms

![alt text](https://github.com/siddhesh1598/HTR_On_Railway_Forms/blob/master/thumbnail.jpg?raw=true)

Performing OCR on Railway Forms to capture user data and update it to the database. The model used is **CRNN model by [Harald Scheidl](https://github.com/githubharald)**. The model is trained on [IAM Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). The dataset contains 1,15,320 isolated and labeled words that have been extracted from pages of scanned text using an automatic segmentation scheme and were verified manually.

![alt text](https://github.com/siddhesh1598/HTR_On_Railway_Forms/blob/master/forms/RailwayForms_8.jpg?raw=true)

Sample Railway Form.


## Technical Concepts

**Word Segmentation:** Scaled Space technique is used to segment words from the sentences in the form. This technique uses an *Anisotropic Gaussian Filter* to generate blobs of words and then these blobs are captured. <br>
More information can be found [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.652.1885&rep=rep1&type=pdf)


**CRNN (Convolutional Recurrent Neural Network) model:** It consists of convolutional NN (CNN) layers, recurrent NN (RNN) layers and a final Connectionist Temporal Classification (CTC) layer. <br>
More information can be found [here](https://arxiv.org/pdf/1507.05717)


**CTC (Connectionist Temporal Classification):** Connectionist temporal classification (CTC) is a type of neural network output and associated scoring function, for training recurrent neural networks (RNNs) such as LSTM networks to tackle sequence problems where the timing is variable. It can be used for tasks like on-line handwriting recognition[1] or recognizing phonemes in speech audio. CTC refers to the outputs and scoring, and is independent of the underlying neural network structure. It was introduced in 2006. <br>
More information can be found [here](https://dl.acm.org/doi/pdf/10.1145/1143844.1143891)


## Getting Started

Clone the project repository to your local machine, then follow up with the steps as required.

### Requirements

After cloning the repository, install the necessary requirements for the project.
```
pip install -r requirements.txt
```

### Training

The maskNet.model file is pre-trained in the images from the [Medical-Mask-Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset). If you wish to train the model from scratch on your own dataset, prepare your dataset in the following way:
1. Load the images in the "*images*" folder
2. Load the xml files containing the bounding box co-ordinates and the labels "bad" or "good" in the "*labels*" folder (please check the xml files from the original dataset to get an idea of the iags used in the files)
3. Store the "*images*" and "*labels*" folder under the "*dataset*" folder

You can then train the model by using the train.py file
```
python train.py --dataset dataset
```
![alt text](https://github.com/siddhesh1598/Face_Mask_Detection/blob/master/plot.png?raw=true)

The plot for Training and Validation Loss and Accuracy.

### Testing

To test the model on your webcam, use the main.py file. 
```
python main.py
```

You can pass some optional parameters for the main.py file:
1. --face: path to face detector model directory <br>
          *default: face_detector*
2. --model: path to trained face mask detector model <br>
          *default: maskNet.model*
3. --confidence: minimum probability to filter weak detections <br>
          *default: 0.35*



## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* Dataset: [Medical-Mask-Dataset](https://www.kaggle.com/vtech6/medical-masks-dataset) <br>
Dataset by **Eden Social Welfare Foundation, Taiwan**. (Re-uploaded by [Mikolaj Witkowski](https://www.kaggle.com/vtech6))

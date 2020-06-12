# HTR_On_Railway_Forms

![alt text](https://github.com/siddhesh1598/HTR_On_Railway_Forms/blob/master/thumbnail.jpg?raw=true)

Performing OCR on Railway Forms to capture user data and update it to the database. The model used is **CRNN model by [Harald Scheidl](https://github.com/githubharald)**. The model is trained on [IAM Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database). The dataset contains 1,15,320 isolated and labeled words that have been extracted from pages of scanned text using an automatic segmentation scheme and were verified manually.

![alt text](https://github.com/siddhesh1598/HTR_On_Railway_Forms/blob/master/System_Architecture.jpg?raw=true)

System Architecture.

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

The model is pre-trained on the words from the IAM dataset. If you wish to train the model on your own dataset, thenn follow the instructions given [here](https://github.com/githubharald/SimpleHTR). Also, the kernel parameters(for word segmentation) are set to work well for the given form design. 

The *main.py* file under the *src* folder containes the main code of the system. It takes an image of the form as an inout and updates the results in the *database.csv* .


## To-Do

The model makes some errors while recognizing the words. It needs to be trained on different tyoes on handwriting, especially on the words which are frequently used while filling a railway form(station names, train numbers, etc).

![alt text](https://github.com/siddhesh1598/HTR_On_Railway_Forms/blob/master/error.png?raw=true)

Incorrectly recognized words.


## Authors

* **Siddhesh Shinde** - *Initial work* - [SiddheshShinde](https://github.com/siddhesh1598)


## Acknowledgments

* Dataset: [IAM Dataset](http://www.fki.inf.unibe.ch/databases/iam-handwriting-database) <br>
* CRNN model: [Harald Scheidl](https://github.com/githubharald) <br>

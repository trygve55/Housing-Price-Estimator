<h1 align="center">Housing Price Guesser</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Heyhey
</p>
<br> 

## 📝 Table of Contents
- [About](#about)
- [Getting Started](#getting_started)
- [Usage](#usage)
- [Built Using](#built_using)
- [Authors](#authors)
- [Acknowledgments](#acknowledgement)

## 🧐 About <a name = "about"></a>
yess

## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Prerequisites
All requirements listed in the 'requirements.txt'-file, simply run the following commands:

```
sudo apt-get install python3.7
sudo apt-get install python3-pip
git clone https://github.com/trygve55/EiT--Gruppe-1
cd EiT--Gruppe-1
python3 -m venv .
source bin/activate
python -m pip install -r requirements.txt
```

Kaggle API setup: https://github.com/Kaggle/kaggle-api.

### Installing

Kaggle Download:

Downloads and extracts all necessary data source files from the Kaggle competition and organizes it into a input directory, ready to use.

```
cd utils
python kaggle_download.py
```
Follow any instructions given as output in case of missing files or directories.

### File Structure

The hierarchy should look like this:

    .
    ├── input                         
    │     ├── data_description.txt
    │     ├── sample_submission.csv
    │     ├── test.csv
    │     ├── train.csv
    │     └── zipped_source
    |          └── ...
    ├── models           
    |     └── model.hdf5
    ├── notebooks                              
    │     └── main.ipynb
    ├── submissions                         
    │     └── submission.csv
    ├── utils                         
    │     └──  kaggle_download.py        
    |
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt


## 🎈 Usage <a name="usage"></a>
yess

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
- [Jupyter Notebook](https://jupyter.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/) 
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    
    
## ✍️ Authors <a name = "authors"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

## 🎉 Acknowledgements <a name = "acknowledgement"></a>
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)

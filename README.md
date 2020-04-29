<h1 align="center">Housing Price Estimator</h1>

<div align="center">

  [![Status](https://img.shields.io/badge/status-active-success.svg)]() 
  [![License](https://img.shields.io/badge/license-MIT-blue.svg)](/LICENSE)

</div>

---

<p align="center">
Housing price estimator for the Norwegian housing market with eXtreme Gradient Boosting and Neural Networks.
</p>
<br> 


## 🏁 Getting Started <a name = "getting_started"></a>
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 


### Prerequisites
This program requires a working install of Python 3.7.

All requirements listed in the 'requirements.txt'-file, simply run the following commands:

```
git clone https://github.com/trygve55/EiT--Gruppe-1
cd EiT--Gruppe-1

#For Linux
python3 -m venv env 
source env/bin/activate

#For Windows
python -m venv env 
source env/Scripts/activate

python -m pip install -r requirements.txt
git clone https://github.com/trygve55/pyfinn
python -m pip install -r pyfinn/requirements.txt
```

### Installing

Simply run the following commands:

```
git clone https://github.com/trygve55/EiT--Gruppe-1
cd EiT--Gruppe-1

#For Linux
python3 -m venv env 
source env/bin/activate

#For Windows
python -m venv env 
source env/Scripts/activate

python -m pip install -r requirements.txt
git clone https://github.com/trygve55/pyfinn
python -m pip install -r pyfinn/requirements.txt
```

### File Structure

The hierarchy should look like this:

    .
    ├── env                              
    │     └── ...
    ├── input                         
    │     ├── data_description.txt
    │     ├── sample_submission.csv
    │     ├── test.csv
    │     └── train.csv
    ├── models           
    |     └── model.hdf5 (added after run)
    ├── notebooks  
    │     ├── finn_data.ipynb                            
    │     └── main.ipynb
    ├── submissions                         
    │     └── submission.csv (added after run)
    ├── utils                         
    │     └── kaggle_download.py        
    |
    ├── .gitignore
    ├── LICENSE
    ├── README.md
    └── requirements.txt


## 🎈 Usage <a name="usage"></a>

#Building dataset

#Training model

#Predicting

## ⛏️ Built Using <a name = "built_using"></a>
- [Python 3.7](https://www.python.org/) 
- [Jupyter Notebook](https://jupyter.org/)
- [TensorFlow 2.0](https://www.tensorflow.org/) 
- [Keras](https://keras.io/)
- [Pandas](https://pandas.pydata.org/)
- [Scikit-learn](https://scikit-learn.org/stable/)
- [XGBoost](https://xgboost.readthedocs.io/en/latest/)
    
    
## ✍️ Authors <a name = "authors"></a>
- Adrian Falla [@Adrianfalla](https://github.com/Adrianfalla)
- Bendik Bogfjellmo [@bendikbo](https://github.com/bendikbo)
- Hans Julian Halvorsen [@hansi77](https://github.com/hansi77)
- Jonas Hjulstad [@jonasbhjulstad](https://github.com/jonasbhjulstad)
- Lars Sandberg [@Sandbergo](https://github.com/Sandbergo)
- Trygve Nerland [@trygve55](https://github.com/trygve55)


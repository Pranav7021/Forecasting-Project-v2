# Forecasting-Project-v2

### How to run the code:

1. Download data from https://www.kaggle.com/datasets/anandhuh/covid19-confirmed-cases-kerala and name file as Covid_19_Data.csv in current directory.

2. Run python train_lstm_cf.py in current directory. The parameters for the trained LSTM and CF models will be saved in lstm.param and cf.param files respectively.

### Motivation for the Convolutional Forecasting (CF) model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. In my previous model, I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. However, this is computationally expensive and might not have been very effective due to the form of the resulting matrix from the product. So, I instead decided to use a 1d convolution on the input sequence to extract patterns.

### Results so far:

For 20 epochs, the LSTM model trained for ~78 seconds. The training loss for the LSTM model was still decreasing. For 20 epochs, the CF model trained for ~2.5 seconds. The training loss for the CF model decreased until epoch 10 and then started increasing. These results are only for the COVID data.

# Forecasting-Project-v2

### Structure of code:

cf.py and lstm.py contain the implementation of my Convolutional Forecasting (CF) and the LSTM models respectively. load_data.py loads the training and testing data. train_model.py implements a train_model function which takes a model and trains it for some number of epochs. train_lstm_cf.py trains both the LSTM and CF models and saves their parameters, using train_model.

### How to run the code:

1. Run python train_lstm_cf.py in current directory. The parameters for the trained LSTM and CF models will be saved in lstm.param and cf.param files respectively.

2. Run python test_lstm_cf.py in current directory. Note: currently, the test is only on the data the models were trained on. I will add testing data shortly.

### Motivation for the CF model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. In my previous model, I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. However, this is computationally expensive and might not have been very effective due to the form of the resulting matrix from the product. So, I instead decided to use a 1d convolution on the input sequence to extract patterns.

### Results so far:

For 40 epochs, the LSTM model trained for ~163 s. The training loss for the LSTM model was still slightly decreasing. For 75 epochs, the CF model trained for ~25 seconds. The training loss for the CF model was still decreasing. These results are only for the COVID data, with training done on CPU only.

#### Sample training output (some epochs omitted):

<img width="472" height="317" alt="Screenshot 2026-02-15 at 11 07 43 AM" src="https://github.com/user-attachments/assets/c23e35b6-07ad-4339-8160-6c7d806cdf0b" />

#### Sample test output:

<img width="468" height="62" alt="Screenshot 2026-02-15 at 11 08 29 AM" src="https://github.com/user-attachments/assets/db90987b-381e-46eb-90ec-9138548a19d0" />

### Data Sources:

COVID data: https://www.kaggle.com/datasets/anandhuh/covid19-confirmed-cases-kerala

# Forecasting-Project-v2

### Structure of code:

cf.py and lstm.py contain the implementation of my Convolutional Forecasting (CF) and the LSTM models respectively. load_data.py loads the training and testing data. train_model.py implements a train_model function which takes a model and trains it for some number of epochs. train_lstm_cf.py trains both the LSTM and CF models and saves their parameters, using train_model.

### How to run the code:

1. Download historical price data for AAPL, NVDA, SPY, GOOGL, and PINS from TradingView.

2. Run python train_lstm_cf.py in current directory. The parameters for the trained LSTM and CF models will be saved in lstm.param and cf.param files respectively.

3. Run python test_lstm_cf.py in current directory.

### Motivation for the CF model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. In my previous model, I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. However, this might not have been very effective due to the somewhat symmetric form of the resulting matrix from the product. In this newer version of the model, I used 1d convolutions which are computationally less expensive, allowing me to use more convolutions to better learn patterns.

### Results so far:

The LSTM trained for 50 epochs in 787 s. The CF trained for 10 epochs in 527 s. The LSTM's total absolute loss on the testing data was ~200 while the CF's was ~301. The CF was more accurate than the LSTM on the testing data 32% of the time. The LSTM was directionally correct on the testing data ~98.8% of the time while the CF was directionally correct 98.5% of the time.

### Data Sources:

Historical price data: TradingView

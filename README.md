# Forecasting-Project-v2

### Structure of code:

cf.py and lstm.py contain the implementation of my Convolutional Forecasting (CF) and the LSTM models respectively. load_data.py loads the training and testing data. train_model.py implements a train_model function which takes a model and trains it for some number of epochs. train_lstm_cf.py trains both the LSTM and CF models and saves their parameters, using train_model.

### How to run the code:

1. Download historical price data for AAPL and NVDA stocks from TradingView.

2. Run python train_lstm_cf.py in current directory. The parameters for the trained LSTM and CF models will be saved in lstm.param and cf.param files respectively.

3. Run python test_lstm_cf.py in current directory. Note: currently, the test is only on the data the models were trained on. I will add testing data shortly.

### Motivation for the CF model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. In my previous model, I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. However, this is computationally expensive and might not have been very effective due to the form of the resulting matrix from the product. So, I instead decided to use a 1d convolution on the input sequence to extract patterns.

### Results so far:

For 100 epochs, the LSTM model trained for ~482 s. For 60 epochs, the CF model trained for ~214 seconds. Training was done on an Apple M4 chip. On stock testing data (780 examples), the LSTM took ~0.35s overall for inference while the CF took ~0.23s. The CF's total absolute loss was 66.8 and the LSTM's total absolute loss was 63.5. The CF was better on ~46.9% of the testing data. 

### Data Sources:

COVID data: https://www.kaggle.com/datasets/anandhuh/covid19-confirmed-cases-kerala

AAPL and NVDA stock data: TradingView

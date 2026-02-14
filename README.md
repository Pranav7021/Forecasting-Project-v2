# Forecasting-Project-v2

## How to run the code:



## Motivation for this model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. In my previous model, I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. However, this is computationally expensive and might not have been very effective due to the form of the resulting matrix from the product. So, I decided to use a 1d convolution instead on the input sequence to extract patterns.

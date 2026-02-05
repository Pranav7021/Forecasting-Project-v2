# Forecasting-Project-v2

Motivation for this model:

Why normalize the data? To eliminate dependence on the units of the data the model is trained on to generalize its predictive ability.

Why convolution? I was inspired by the use of convolutions in image recognition to help the model detect various patterns. I wanted to help the model learn intertemporal patterns on matrices obtained by taking the product of 50\*1 and 1*50 matrices representing the previous 50 observations and its transpose. Using convolutions of different sizes, I hoped the model would learn patterns that hold for different time frames.

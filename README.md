Introduction
============

Forecasting stock prices plays an important role in setting a trading
strategy or determining the appropriate timing for buying or selling a
stock. In this project, we use a model, called feature fusion long
short-term memory-convolutional neural network (LSTM-CNN) model. It
combines features learned from different representations of the same
data, namely, stock time series and stock chart images, to predict stock
prices.

Related Work
============

Stock prediction is one of the most challenging and long standing
problems in the field of time series data. H.Q.Thang @hust used Gaussian
Process Regression and Autoregressive Moving Average Model to predict
Vietnam Stock Index Trend. N.V.Son @vbd used ARIMA and LSTM to predict
some stock symbols like APPL (Apple), AMZN (Amazon).

In this project, we use a combined model called long short-term
memory-convolutional neural network (LSTM-CNN) to predict closed price
of Dow Jones Industrial Average (DJIA). As an extension, the model will
be implemented on VN-30 index data. Kim T, Kim HY @ours implemented
fusion LSTM-CNN model on 2018-2019 S&P 500 data. Simiarly, Hao Y, Gao Q
constructed LSTM-CNN model using 2009-2019 S&P 500 data.

Expected Results
================

-   Understand CNN, LSTM model and its application to time series
    forecasting problems.

-   Understand forecasting stock prices problem, the application of
    machine learning in this field and the shortcomings of using them in
    the real market.

Reference
=========

<span>1.</span> H.Q.Thang. *Vietnam Stock Index Trend Prediction using
Gaussian Process Regression and Autoregressive Moving Average Model*.
Research and Development on Information and Communication Technology,
HUST, 2018.

<span>2.</span> Kim T, Kim HY. *Forecasting stock prices with a feature fusion LSTM-CNN
model using different representations of the same data*. PLoS ONE 14(2):
e0212320, 2019.
<https://doi.org/10.1371/journal.pone.0212320>

<span>3.</span> Hao Y, Gao Q. *Predicting the Trend of Stock Market Index Using the
Hybrid Neural Network Based on Multiple Time Scale Machine Learning*.
MDPI Appl. Sci. 2020, 10(11), 3961.
<https://doi.org/10.3390/app10113961>
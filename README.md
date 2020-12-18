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

Project Planning
================

-   Get historical data of DJI on Yahoo! Finance.

-   Explore, transform data.

-   Build CNN, LSTM model independently, then combine them.

-   Evaluation on DJI, VN-30 index and some popular symbols.

| Time            | Tasks                              | Owner                         |
|-----------------|------------------------------------|-------------------------------|
| Dec 17 - Dec 18 | Get historical data                | M.H.Lan                       |
|                 | Research methods                   | M.H.Lan, H.T.Hoai, N.T.H.Phuc |
| Dec 19 - Dec 22 | Explore and transform data         | H.T.Hoai                      |
|                 | Implement CNN model                | N.T.H.Phuc                    |
|                 | Implement LSTM model               | M.H.Lan                       |
|                 | Implement CNN-LSTM                 | H.T.Hoai                      |
| Dec 22 - Dec 24 | Fine-tuning and evaluation         | H.T.Hoai, M.H.Lan, N.T.H.Phuc |
| Dec 25 - Dec 27 | Write report                       | H.T.Hoai, M.H.Lan, N.T.H.Phuc |
|                 | Deploy web app (given enough time) | H.T.Hoai, M.H.Lan, N.T.H.Phuc |

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

<span>2.</span> N.V.Son, N.T.Hien, D.Q.Khai, P.N.Dong. *Stock Prediction with ARIMA and
LSTM*. Vingroup Big Data Institute, 2020.

<span>3.</span> Kim T, Kim HY. *Forecasting stock prices with a feature fusion LSTM-CNN
model using different representations of the same data*. PLoS ONE 14(2):
e0212320, 2019.\
<https://doi.org/10.1371/journal.pone.0212320>

<span>4.</span> Hao Y, Gao Q. *Predicting the Trend of Stock Market Index Using the
Hybrid Neural Network Based on Multiple Time Scale Machine Learning*.
MDPI Appl. Sci. 2020, 10(11), 3961.\
<https://doi.org/10.3390/app10113961>

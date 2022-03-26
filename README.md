# CNN-BiLSTM-network

CNN-Bidirectional LSTM network to forecast long term traffic flow in Madrid. 

## Introduction
In this repository we present the code implemented to forecast long-term traffic flow in four stations of Madrid by applying a hybrid model which combines a Convolutional block and a Bidirectional LSTM block. The first block is composed by a convolutional neural netowrk, a maxpooling layer and a flatten layer. The second one, is composed by a bidirectional LSTM network, a dropout layer and a dense layer. We also include in this repository all the datasets with the predictor variables and the target variable. The predictor variables are summarised in the final section of this file. 



## How can you apply the model?
You can dowload the [MainFile](https://github.com/MMH1997/CNN-BiLSTM-network/blob/main/Main.ipynb) (in any form). This file develops the CNN-BiLSTM proposed model and makes the forecasting only in one station and one granularity: 12 hours, C/Arturo Soria.

You can also dowload data from other stations or granularities. 

The two modifications that you need to do in the [MainFile](https://github.com/MMH1997/CNN-BiLSTM-network/blob/main/Main.ipynb) to work with other datasets are:

First of all, change the name of the data file, exactly in 
    
`data=pd.read_csv('AS2(t-12).csv')`
    
Secondly, you need to change the way 'hour' variable is created. For it, you can check the file named [HourVariable](https://github.com/MMH1997/CNN-BiLSTM-network/blob/main/HourVariable.ipynb) and, depending on the granularity, you can employ the needed way to create the variable. You just must change the following lines in [MainFile](https://github.com/MMH1997/CNN-BiLSTM-network/blob/main/Main.ipynb):
    
`a1=list(range(13,24))`

`a2=list(range(0,24))*1306`

`a3=list(range(0,24))`

`a4=a1+a2+a3`


## Summary of the predictor variables.
* Auxiliary station (1) and Auxiliary station(2): Traffic flow in two near stations to the target one. 
* Target station: Traffic flow in target station.
* Tmax, tmin, tmed: Maximum, minimum and average temperature (of the day)
* Rainfall: Daily rain.
* Type of day: Working day, public holiday or weekend.
* Hour.

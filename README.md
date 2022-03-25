# CNN-BiLSTM-network
CNN-Bidirectional LSTM network to forecast long term traffic flow in Madrid. 

You can dowload the 'Main' file (in all its forms). This file develops the CNN-BiLSTM proposed model and makes the forecasting only in one station and one granularity: 12 hours, C/Arturo Soria.

You can also dowload data from other stations or granularities. 

The two modifications that you need to do in the 'Main' file to work with other datasets are:
    -First of all, change the name of the data file, exactly in 'data=pd.read_csv('AS2(t-12).csv')'. 
    -Secondly, you need to change the way 'hour' variable is created. For it, you can check the file named 'HourVariable' and, depending on the granularity, you can employ the     needed way to create the variable. You just must change the following lines:
    a1=list(range(13,24))
    a2=list(range(0,24))*1306
    a3=list(range(0,24))
    a4=a1+a2+a3

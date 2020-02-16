Aim:
Predicts Stock prices of S&P 500 companies by using Analyst reports called 8-k filings. Sentiment prediction is done using only sentence and without numerical metrics. 

Techniques used:
1) A Bi-LSTM module was developed and used for dependency parsing. The features extracted was fed to seperate CNN,RNN,MLP and CNN-RNN modules. 
2) In our experiment we found CNN-RNN to perform the best.

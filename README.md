## LSTMSentimentAnalytics
Sentiment Analysis for Movie review. Using LSTM (RNN Neural Network). 
### Description
Objective is to build machine learning algorithm to predict sentiment of review done on movie.
We have 25000 already tagged data, and model is train based on tagged data.
I have used LSTM to predict sentiment of review. LSTM will not only consider words but also sequence of word in review.
Since Structure of language is having significance. LSTM, will take care of sequence of word in statement. 

### Tools and Technique

- `Platform`  :: python
- `Framework` :: keras
- `Technique` :: one hot encoding. Vocabulary is defined and each word is allocated to unique id. Review is devided in words and each word is converted to number ( id of word in dictionary of Vocabulary).
- `Algorithm` :: LSTM With 1 layer 300 units. 

### Data

- `Train` :: 25000
- `Positive tagged data` ::  15000
- `Negetive tagged data` ::  15000
- `Test` :: 25000
- `Source` :: in built dataset in keras

### Files

- `SentimentAnalysisLSTM.py` :: Coding file for step by step process with details. 

### Conclusion

Model is train and tested with efficiency of 88% on test data.






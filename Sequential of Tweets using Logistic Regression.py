#Import the NLTK library
import nltk
from os import getcwd

nltk.download('twitter_samples')
nltk.download('stopwords')

#Scientific Calculation and Data Visualization
import numpy as np
import pandas as pd
from nltk.corpus import twitter_samples
from utils import build_freqs,process_tweet,extract_features,predict_tweet,sigmoid,test_logistic_regression

#Selection of the positive and negative tweets
positive_tweets = twitter_samples.strings('positive_tweets.json')
negative_tweets = twitter_samples.strings('negative_tweets.json')

#Dividing the data into two parts (Training set - 80%) and (Testing set - 20%)
test_pos = positive_tweets[4000:]
train_pos =positive_tweets[:4000]
test_neg = negative_tweets[4000:]
train_neg = negative_tweets[:4000]

train_x = train_pos + train_neg 
test_x = test_pos + test_neg

#Combining the x and y lables 
train_y = np.append(np.ones((len(train_pos), 1)), np.zeros((len(train_neg), 1)), axis=0)
test_y = np.append(np.ones((len(test_pos), 1)), np.zeros((len(test_neg), 1)), axis=0)

#Printing the Shape Train and Test Sets
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

#create the frequency dictionaries
freqs = build_freqs.build_freqs(train_x,train_y)

#printing out the output
print("train_y.shape = " + str(train_y.shape))
print("test_y.shape = " + str(test_y.shape))

# testing the function below
print('This is an example of a positive tweet: \n', train_x[0])
print('\nThis is an example of the processed version of the tweet: \n', process_tweet.process_tweet(train_x[0]))

# Testing the signmoid function 

if (sigmoid(0) == 0.5):
    print('Happy!')
else:
    print('Sad!')

if (sigmoid(4.92) == 0.9927537604041685):
    print('Cool!')
else:
    print('Sadness Again!')
    
#Use of the cost function along with Logistic Regression
#verification of the model when it is being predicting to 1; houwever, the actual label is 0. In that case, the loss becomes a very large positive number
-1 * (1 - 0) * np.log(1 - 0.9999)


#verification of the model when it is being predicting to 0; houwever, the actual label is 1. In that case, the loss becomes a very large positive number
-1  * np.log(0.0001)


# Checking the Gradient Descent Function
# Construct a synthetic test case using numpy PRNG functions
np.random.seed(1)
# X input is 10 x 3 with ones for the bias terms
tmp_X = np.append(np.ones((10, 1)), np.random.rand(10, 2) * 2000, axis=1)
# Y Labels are 10 x 1
tmp_Y = (np.random.rand(10, 1) > 0.35).astype(float)

# Applying the gradient descent
tmp_J, tmp_theta = gradientDescent(tmp_X, tmp_Y, np.zeros((3, 1)), 1e-8, 700)
print(f"The cost after training is {tmp_J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(tmp_theta)]}")

# test on training data
tmp1 = extract_features.extract_features(train_x[0], freqs)
print(tmp1)

# checking for when the words are not in the freqs dictionary
tmp2 = extract_features.extract_features('blorb bleeeeb bloooob', freqs)
print(tmp2)

#Training of the Logistic Regression Model
# collect the features 'x' and stack them into a matrix 'X'
X = np.zeros((len(train_x), 3))
for i in range(len(train_x)):
    X[i, :]= extract_features.extract_features(train_x[i], freqs)

# training labels corresponding to X
Y = train_y

# Applying the gradient descent
J, theta = gradientDescent(X, Y, np.zeros((3, 1)), 1e-9, 1500)
print(f"The cost after training is {J:.8f}.")
print(f"The resulting vector of weights is {[round(t, 8) for t in np.squeeze(theta)]}")

# Final checking for the prediction test
for tweet in ['I am happy', 'I am bad', 'this movie should have been great.', 'great', 'great great', 'great great great', 'great great great great']:
    print( '%s -> %f' % (tweet, predict_tweet.predict_tweet(tweet, freqs, theta)))


my_tweet = 'I am learning :)'
predict_tweet.predict_tweet(my_tweet, freqs, theta)


tmp_accuracy = test_logistic_regression.test_logistic_regression(test_x, test_y, freqs, theta)
print(f"Logistic regression model's accuracy = {tmp_accuracy:.4f}")


#Error Analysis for the Model 
print('Label Predicted Tweet')
for x,y in zip(test_x,test_y):
    y_hat = predict_tweet.predict_tweet(x, freqs, theta)

    if np.abs(y - (y_hat > 0.5)) > 0:
        print('THE TWEET IS:', x)
        print('THE PROCESSED TWEET IS:', process_tweet.process_tweet(x))
        print('%d\t%0.8f\t%s' % (y, y_hat, ' '.join(process_tweet.process_tweet(x)).encode('ascii', 'ignore')))
        
my_tweet = 'I am going to America'
print(process_tweet.process_tweet(my_tweet))
y_hat = predict_tweet.predict_tweet(my_tweet, freqs, theta)
print(y_hat)
if y_hat > 0.5:
    print('Positive sentiment')
else: 
    print('Negative sentiment')    

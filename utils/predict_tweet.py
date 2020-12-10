from utils import extract_features,sigmoid
import numpy as np

def predict_tweet(tweet, freqs, theta):
   
    # extract the features of the tweet and store it into x
    x = extract_features.extract_features(tweet,freqs)
    
    # make the prediction using x and theta
    y_pred = sigmoid.sigmoid(np.dot(x,theta))
      
    return y_pred

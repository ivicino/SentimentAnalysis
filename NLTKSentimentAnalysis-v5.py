# import libraries
import pandas as pd
import numpy as np

import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download('wordnet')

# make a list of negative and positive words... Then count how many negative words there are and positive words and 
# then subtract the negative words from the positive. 
# If the total is positive, it will be marked as a positive comment, else a negative comment.
negwords = ['would be', 'issues', 'more training', 'problems', 'causing', 'problem', 'late', 'rude', 'argumentative', 'argue', 'negative']    # add to this list if required
pattern = '|'.join(negwords)
# print(pattern)
poswords = ['great', 'amazing', 'incredible', 'informative', 'fun', 'thorough', 'personalized', 'fun', 'in-depth', 'positive']
pospattern = '|'.join(poswords)


# Load the data
df = pd.read_csv(".\Documents\Data\sciPOP\sciPOP 2023.csv")

improve = df['Is there anything further you would like our team to be aware of? (Improvements, questions, comments, concerns?)']
# print(improve)
improvedf = improve.dropna(axis=0)
# print(improvedf)

def preprocess_text(text):
    # tokenize the dataframe
    word = word_tokenize(text)

    # get rid of punctuation from df and lowercase it
    nopunc = []
    for w in word:
        if w.isalpha():
            nopunc.append(w.lower())

    # Join the tokens back into a string
    processed_text = ' '.join(nopunc)
    return processed_text


# apply the function df
improvedf = improvedf.apply(preprocess_text)
# print(improvedf)

# print('==================improvedDF======================', '\n', improvedf)

# check if entry contains negative words
# flagsdf = improvedf.str.contains(pattern, regex=True)    # ~ means not or negation..., regex = True because the pattern is a regex...
# flagsdf will output True if the responce is negative
 
flagsdfcount = improvedf.str.count(pattern)
# print(flagsdfcount)

# flagsdf = ~improvedf.str.contains(pattern, regex=True)    # ~ means not or negation..., regex = True because the pattern is a regex...
# # flagsdf will output False if the responce is negative 
# print('Negatory \n', flagsdf, '\n')

# # check if entry contains Positive words
# posflagsdf = improvedf.str.contains(pospattern, regex=True)    # ~ means not or negation..., regex = True because the pattern is a regex...
# print('Positive \n', posflagsdf, '\n')

posflagsdfcount = improvedf.str.count(pospattern)
# print(posflagsdfcount)


flagsdf_new = flagsdfcount.copy()
# flagsdf_new = flagsdf_new.map({True: 'True', False: 'False'})   # convert booleans to strings...

posflagsdf_new = posflagsdfcount.copy()
# posflagsdf_new = posflagsdf_new.map({True: 'True', False: 'False'})   # convert booleans to strings...


subdf = posflagsdf_new.subtract(flagsdf_new)
# print(subdf)

for i, entry in enumerate(subdf):
    # print(i)
    if entry > 0:
        subdf.replace(subdf.iloc[i], "Positive comment (+)", inplace=True)
    elif entry < 0:
        subdf.replace(subdf.iloc[i], "Negative comment (-)", inplace=True)
    elif entry == 0:
        subdf.replace(subdf.iloc[i], "Neurtral comment (0)", inplace=True)

    

# print(subdf)
df['Sentiment'] = subdf
print(df)

print('saving...')
df.to_csv('POP2023-sentiment-v5.csv')


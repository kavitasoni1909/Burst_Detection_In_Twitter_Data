#!/usr/bin/env python
# coding: utf-8

# In[143]:


import numpy as np
import scipy.stats as stats
from collections import Counter
from itertools import dropwhile
import string
import re

import burst_detection as bd

import seaborn as sns
import matplotlib.pyplot as plt  
get_ipython().run_line_magic('matplotlib', 'inline')


# In[144]:


import pandas as pd


# # format plots

# In[145]:



#format plots
sns.set(style='white', context='notebook', font_scale=1.5, 
        rc={'font.sans-serif': 'DejaVu Sans', 'lines.linewidth': 2.5})

#create a custom color palette
palette21 = ['#21618C', '#3498DB', '#AED6F1', '#00838F', '#00BFA5',
             '#F1C40F', '#F9E79F', '#E67E22', '#922B21', '#C0392B', 
             '#E6B0AA', '#6A1B9A', '#8E44AD', '#D7BDE2', '#196F3D', 
             '#4CAF50', '#A9DFBF', '#4527A0', '#7986CB', '#555555', 
             '#CCCCCC']
sns.palplot(palette21)

#create a color map
blog_blue = '#64C0C0'
blue_cmap = sns.light_palette(blog_blue, as_cmap=True)


# # Load data and preprocess titles

# In[146]:


tweets = pd.read_csv('tweets.csv', delimiter=';', nrows=10000, lineterminator='\n' )
tweets.head()


# In[147]:


# Rename columns
tweets.columns = ["id", "user", "fullname", "url", "timestamp", "replies","likes","retweets","text"]


# In[148]:


tweets.head()


# In[149]:


#preprocess words in title: split words, convert to lowercase, and remove punctuation
tweets['words'] = tweets['text'].apply(lambda x: x.lower().split())
tweets['words'] = tweets['words'].apply(lambda x: [word.strip(string.punctuation) for word in x])


# In[150]:


tweets['date'] = pd.to_datetime(tweets['timestamp'],format= '%Y-%m-%d').dt.date


# In[151]:


#add columns for year and month
tweets['year'] = tweets['date'].apply(lambda x: x.year)
tweets['month'] = tweets['date'].apply(lambda x: x.month)


# In[152]:


tweets.head()


# In[153]:


tweets = tweets[['words', 'date', 'year', 'month', 'timestamp']]


# In[154]:


tweets.head()


# In[155]:


#count all words in the tweets
word_counts = Counter(tweets['words'].apply(pd.Series).stack())
print('Number of unique words: ',len(word_counts))


# In[156]:


#remove words that appear fewer than X times
count_threshold = 50

for key, count in dropwhile(lambda x: x[1] >= count_threshold, word_counts.most_common()):
    del word_counts[key]
print('Number of unique words with at least',count_threshold,'occurances: ',len(word_counts))


# In[157]:


#create a list of unique words
unique_words = list(word_counts.keys())
unique_words[:10]


# In[158]:


#count the number of tweets published each month
d = tweets.groupby(['year','month'])['words'].count().reset_index(drop=True)
print(d)


# In[159]:


#plot the number of tweets posted each month

#initialize a figure
plt.figure(figsize=(10,5))

#plot bars
#axes = plt.bar(d.index, d, width=1, color=blue_cmap((d-np.min(d))/(np.max(d)-np.min(d)))) #color according to height
axes = plt.bar(d.index, d, width=1, color=blue_cmap(d.index.values/d.index.max()))  #color according to month

#format plot
plt.grid(axis='y')
plt.xlim(0,len(d))
plt.xticks(range(0,len(d),24), range(2019,2019,5), rotation='vertical')
plt.tick_params(axis='x', length=5)
plt.title('Number of Bitcoin tweets posted in 2019-05')
sns.despine(left=True)

plt.tight_layout()
plt.savefig('bitcoin_tweets_over_time.png', dpi=300)


# In[160]:


all_r = pd.DataFrame(columns=unique_words, index=d.index)

for i, word in enumerate(unique_words):
    
    all_r[word] = pd.concat([tweets.loc[:,['year','month']], 
                             tweets['words'].apply(lambda x: word in x)], 
                            axis=1) \
                    .groupby(by=['year','month']) \
                    .sum() \
                    .reset_index(drop=True)
                
    #print out a status indicator
    if np.mod(i,100)==0:
        print('word',i,'complete')
    
all_r


# In[161]:


def plot_most_common_words(word_counts, n, title, gradient, label_type):
    
    #filter stop words
    discard_words = ['of','in','and','the','a','with','for','to','on','an','by','using',
                     'from','as','is','at','between','during', 'you', 'this'
                     'now','will','that','are','your','it']
    for key in discard_words:
        del word_counts[key]
    word_counts = pd.DataFrame(word_counts.most_common()[:n], columns=['word','count'])

    #define colors for bars
    if gradient:
        bar_colors = blue_cmap((word_counts['count'])/(word_counts['count'].max()))
    else:
        bar_colors = blog_blue

    #create a horizontal bar plot
    plt.barh(range(n,0,-1), word_counts['count'], height=0.85, color=bar_colors, alpha=1)

    #format plot
    sns.despine(left=True,bottom=True)
    plt.ylim(0,n+1)
    plt.title(title)
    plt.grid(axis='x')

    #label bars
    if label_type == 'counts':
        plt.yticks(range(n,0,-1), word_counts['word']);
        for i, row in word_counts.iterrows():
            plt.text(row['count']-100,50-i-0.2, row['count'], horizontalalignment='right', fontsize='12', color='white')

    elif label_type == 'labeled_bars_left':
        plt.yticks(range(n,0,-1), []);
        for i, row in word_counts.iterrows():
            plt.text(50,n-i-0.2, row['word'], horizontalalignment='left', fontsize='14')

    elif label_type == 'labeled_bars_right':
        plt.yticks(range(n,0,-1), []);
        for i, row in word_counts.iterrows():
            plt.text(row['count'],n-i-0.2,row['word'], horizontalalignment='right', fontsize='14')

    else:
        plt.yticks(range(n,0,-1), word_counts['word']);


# In[162]:


plt.figure(figsize=(10,15))
plot_most_common_words(word_counts=word_counts, n=50, 
                       title='Top 50 words in tweets', 
                       gradient=False, label_type='counts')
plt.tight_layout()
plt.savefig('most_common_words_in_tweets.png',dpi=300)


# # Find bursts for each unique word

# In[168]:


#find bursts

#create a dataframe to hold results
all_bursts = pd.DataFrame(columns=['begin','end','weight'])

#define variables
s = 2         #resolution of state jumps; higher s --> fewer but stronger bursts
gam = 0.5     #difficulty of moving up a state; larger gamma --> harder to move up states, less bursty
n = len(d)    #number of timepoints

#loop through unique words
for i, word in enumerate(unique_words):

    r = all_r.loc[:,word].astype(int)

    #find the optimal state sequence (using the Viterbi algorithm)
    [q, _, _, p] = bd.burst_detection(r,d,n,s,gam,smooth_win=5)

    #enumerate the bursts
    bursts = bd.enumerate_bursts(q, word)

    #find weight of each burst
    bursts = bd.burst_weights(bursts, r, d, p)
    #add the bursts to a list of all bursts
    all_bursts = all_bursts.append(bursts, ignore_index=True)
    
    #print a progress report every 100 words
    if np.mod(i,100)==0:
        print('word',i,'complete')

all_bursts.sort_values(by='weight', ascending=False)
#scroll down the warning to see the bursts of each word


# In[ ]:





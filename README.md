# Introduction
Sentiment Analysis (opninion analysis, or opinion mining) is used to predict their sentiment of opinion from textual data, which is usually unstructured text. The output of this analysis is a polarity which is 0 for neutral statements, > 0 for positie statements, and < 0 for negative statements. Here, we have classified sentiments to positive and negative classes using supervised learning. Here, movie reviews have been analysed to predict the sentiment.

**Dataset** <br>
The movie reviews were from <a href = "https://ai.stanford.edu/~amaas/data/sentiment/">Internet Movie Database (IMDB)</a>, which was originally proposed by  Mass <i>et al.</i> in their paper on <a href="https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf">Learning Word Vectors for Sentiment Analysis</a>, and contains 50000 labelled movie reviews. However, here, we have used web curling nand extracted data from a <a href = "https://github.com/dipanjanS/text-analytics-with-python/blob/master/Old-First-Edition/source_code/Ch07_Semantic_and_Sentiment_Analysis/movie_reviews.csv">GitHub repository</a>. <br>
<br>
A sample of the dataset can be seen below.

<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Dataset.PNG" width = 500>

# Libraries Used
1. numpy <br>
2. pandas <br>
3. nltk (Natural Language Toolkit)<br>
4. bs4 <br>
5. sklearn <br>

# Text Pre-processing
The data has first been pre-processed to reduce noise and build more meaningful features. This has been achieved using the following steps-
1. Remove hyperlinks and hashtags - Implemented using regular expressions  **INCLUDE SELF CODE LINK HERE** <br>
2. Remove accented characters - Changed acccented characters to ASCII charcters <br>
3. Remove punctuation <br>
4. Change all the chanracters to lowercase <br>
5. Stemming- Affixes (prefixes and suffixes) are removed to extract the base words. This stemmed word may or may not be lexicographically correct. This has been done using Porter Stemming. For example- <br>
flying --> fly<br>
beautiful --> beauti<br>
6. Remove stopwords - Stopwords are the words which has little significance in text understanding. Such as 'a', 'an', and so on. Below is a list of all the stopwords included in nltk for the English language.

<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Stopwords.PNG" width = 1000>

# Feature Engineering
The unstructured text data has to be converted to appropriate features in order to apply machine learning algorithms. Accordingly, we have used<br>
1. **Bag of Words** (1-gram) - The model disregards word order and sequences, and directly represents the frequency of a specific word in each text document.<br>
2. **Bag of N-Grams** - This is similar to Bag of Words, except that the frequency is calculated for a sequence of words instead of a single word. Here, 'N' represents the number of words in the sequence.<br>
3. **TF-IDF** (Term Frequency-Inverse Document Frequency) - As the above two methods give more importance to frequency for words, they may give more weightage to frequently occuring words which may not carry much information, such as 'a', and 'the'. On the other hand, words which occur infrequently may carry more information and may serve as more important features. Accordingly, the term frequency (word frequency in each document) is multiplied by the inverse document frequency (inverse of document frequency for each word) to obtain the TF-IDF representation.<br>

# Classification using Logistic Regression
1. **Bag of Words** <br>
<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Logistic_BagofWords.PNG" width = 500>

2. **TF-IDF** <br>
<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Logistic_TFIDF.PNG" width = 500>

# Classification using Support Vector Machine
1. **Bag of Words** <br>
<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Support_BagofWords.PNG" width = 500>

2. **TF-IDF** <br>
<img src = "https://github.com/isha-git/Sentiment-Analysis-Using-Supervised-Learning/blob/master/images/Support_TFIDF.PNG" width = 500>

# References
The <a href = "https://www.apress.com/gp/book/9781484243534">book</a> and <a href = "https://github.com/dipanjanS/text-analytics-with-python">repository</a> provided by Dipanjan Sarkar were followed for this analysis, along with some more resources such as <a href = "https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/">GeeksforGeeks</a>, Stack Overflow, and the <a href = "https://scikit-learn.org/stable/">sklearn documentation</a>.

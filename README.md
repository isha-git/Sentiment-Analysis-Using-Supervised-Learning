# Introduction
Sentiment Analysis (opninion analysis, or opinion mining) is used to predict their sentiment of opinion from textual data, which is usually unstructured text. The output of this analysis is a polarity which is 0 for neutral statements, > 0 for positie statements, and < 0 for negative statements. Here, we have predicted sentiments using supervised learning. Here, movie reviews have been analysed to predict the sentiment.

**Dataset** <br>
The movie reviews were from <a href = "https://ai.stanford.edu/~amaas/data/sentiment/">Internet Movie Database (IMDB)</a>, which was originally proposed by  Mass <i>et al.</i> in their paper on <a href="https://ai.stanford.edu/~amaas/papers/wvSent_acl2011.pdf">Learning Word Vectors for Sentiment Analysis</a>. However, here, we have used web curling and extracted data from a <a href = "https://github.com/dipanjanS/text-analytics-with-python/blob/master/Old-First-Edition/source_code/Ch07_Semantic_and_Sentiment_Analysis/movie_reviews.csv">GitHub repository</a>.

IMAGE

# Libraries Used
1. numpy <br>
2. pandas <br>
3. nltk <br>
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
6. Remove stopwords - Stopwords are the words which has little significance in text understanding. Such as 'a', 'an', and so on. **INCLUDE IMAGE OF ALL STOPWORDS**

# Methods 

# References
The <a href = "https://www.apress.com/gp/book/9781484243534">book</a> and <a href = "https://github.com/dipanjanS/text-analytics-with-python">repository</a> provided by Dipanjan Sarkar were followed for this analysis, along with some more resources such as <a href = "https://www.geeksforgeeks.org/text-preprocessing-in-python-set-1/">GeeksforGeeks</a>, Stack Overflow, and the <a href = "https://scikit-learn.org/stable/">sklearn documentation</a>.

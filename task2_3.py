#!/usr/bin/env python
# coding: utf-8

# # Assignment 2: Milestone I Natural Language Processing
# ## Task 2&3
# #### Student Name: Pallavi Kollipara
# #### Student ID: s4015344
# 
# Date: 11-05-2024
# 
# Version: 1.0
# 
# Environment: Python 3 and Jupyter notebook
# 
# Libraries used: please include all the libraries you used in your assignment, e.g.,:
# * pandas
# * re
# * numpy
# 
# ## Introduction
# 
# The introduction and requirements for both Task 2 and Task 3 along with the necessity of loading `vocab.txt` and `processed_job_ads.txt`.
# 
# **Task 2: Job Advertisement Description Processing**
# 
# Task 2 involves processing job advertisement descriptions to generate feature representations for machine learning models. The goal is to extract meaningful information from the text data to facilitate classification tasks. Key steps include:
# 
# 1. **Vocabulary Generation**: Extracting a vocabulary from the job advertisement descriptions. This involves identifying unique words and assigning indices to them. The vocabulary serves as a basis for feature representation.
# 
# 2. **Feature Representation**: Generating feature representations for the job advertisement descriptions. This can be done using techniques like Term Frequency-Inverse Document Frequency (TF-IDF) or Word Embeddings. These representations capture the semantic meaning and importance of words within the descriptions.
# 
# **Task 3: Job Advertisement Classification**
# 
# Task 3 focuses on building machine learning models for classifying job advertisement categories based on their textual content. The task involves comparing different language models and exploring the impact of additional information (such as job titles) on classification accuracy. Key steps include:
# 
# 1. **Model Comparison**: Experimenting with different language models (such as TF-IDF and Word Embeddings) to determine which performs best for classification tasks.
# 
# 2. **Incorporating Additional Information**: Investigating whether including additional information, such as job titles, improves classification accuracy. This involves comparing models trained with only job descriptions, only job titles, and both combined.
# 
# **Importance of Loading `vocab.txt` and `processed_job_ads.txt`**
# 
# For both Task 2 and Task 3, it's essential to load two important files:
# 
# 1. **vocab.txt**: This file contains the vocabulary generated from the job advertisement descriptions in Task 2. It includes unique words and their corresponding indices, necessary for feature representation.
# 
# 2. **processed_job_ads.txt**: This file contains preprocessed job advertisement data, including titles, descriptions, categories, web indices, and company names. Loading this file provides access to the actual job advertisement texts, crucial for generating document representations and conducting classification tasks.
# 
# By loading and processing these files, we ensure that we have the necessary data and vocabulary required for conducting experiments and building machine learning models effectively for job advertisement processing and classification.

# ## Importing libraries 

# In[1]:


import pandas as pd
import numpy as np 
import re
import gensim.downloader as api
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import KFold
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score


# In[2]:


vocab = {}

with open('vocab.txt', 'r') as f:
    for line in f:
        word, index = line.strip().split(':')
        vocab[word] = int(index)

vocab


# In[3]:


job_ads = []

with open('processed_job_ads.txt', 'r') as f:
    current_job = {}
    for line in f:
        line = line.strip()
        if line.startswith("Title:"):
            if current_job:
                job_ads.append(current_job)
                current_job = {}
            current_job['title'] = line[len("Title:"):].strip()
        elif line.startswith("Category:"):
            current_job['Category'] = line[len("Category:"):].strip()
        elif line.startswith("Webindex:"):
            current_job['webindex'] = line[len("Webindex:"):].strip()
        elif line.startswith("Company:"):
            current_job['company'] = line[len("Company:"):].strip()
        elif line.startswith("Description:"):
            current_job['description'] = line[len("Description:"):].strip()

    if current_job:
        job_ads.append(current_job)



# In[4]:


descriptions = [job['description'] for job in job_ads]


# In[5]:


tokenized_descriptions = [description.split() for description in descriptions]


# ## Task 2. Generating Feature Representations for Job Advertisement Descriptions

# This task is to generate different types of feature representations for the collection of job advertisement descriptions. These feature representations will be used for various natural language processing tasks, such as text classification, clustering, or information retrieval.
# 
# The task involves generating the following feature representations:
# 
# 1. **Bag-of-Words Model (Count Vector Representation)**
#    - You need to generate the count vector representation for each job advertisement description.
#    - The count vector representation is a sparse vector where each element corresponds to a word in the vocabulary (generated in Task 1).
#    - The value of each element is the count of the corresponding word in the job advertisement description.
#    - The count vectors should be saved in the `count_vectors.txt` file, following the specified format.
# 
# 2. **Word Embedding Models**
#    - You need to generate feature representations based on pre-trained word embedding models, such as FastText, Word2Vec (GoogleNews-300), or GloVe.
#    - For each job advertisement description, you need to create two types of document embeddings:
#      - **TF-IDF Weighted Document Embedding**: In this representation, the word embeddings are weighted by the Term Frequency-Inverse Document Frequency (TF-IDF) values of the corresponding words.
#      - **Unweighted Document Embedding**: In this representation, the word embeddings are averaged without any weighting.

# ### Bag-of-words model:
# 
# The bag-of-words model is a simple technique for representing text data as numerical feature vectors. It involves counting the occurrences of each word in a document and using those counts as features. Here are the steps to generate the count vector representation for job advertisement descriptions using the bag-of-words model:
# 
# 1. Load the vocabulary from the `vocab.txt` file generated in Task 1. This will be the set of unique words (features) used to represent each document.
# 
# 2. For each job advertisement description:
#    - Tokenize the text into individual words
#    - Count the occurrences of each word from the vocabulary in the tokenized text
#    - Create a vector with length equal to the vocabulary size, where each element corresponds to the count of the respective word
# 
# 3. Save the count vectors for all job advertisement descriptions in the `count_vectors.txt` file using the specified format:
#    - Start each line with `#` followed by the `webindex` of the job advertisement and a comma `,`
#    - Represent the count vector as a comma-separated list of `word_index:word_count` pairs
#    - `word_index` is the index of the word in the vocabulary (0-based indexing)
#    - `word_count` is the number of occurrences of that word in the document

# In[6]:


def generate_count_vector(description):
    count_vector = {}
    for word in description.split():
        if word in vocab:
            index = vocab[word]
            count_vector[index] = count_vector.get(index, 0) + 1
    return count_vector


# ### Models based on word embeddings:
# 
# this task is related to generating feature representations based on word embeddings.
# 
# For this part of the task, we need to choose one pre-trained word embedding model, such as FastText, Word2Vec (GoogleNews-300), or GloVe. And I have loaded the Word2Vec (GoogleNews-300) model.
# 
# Using the chosen word embedding model (Word2Vec in your case),generate two types of document embeddings for each job advertisement description:
# 
# 1. **TF-IDF Weighted Document Embedding**:
#    - In this representation, you need to weight the word embeddings by the Term Frequency-Inverse Document Frequency (TF-IDF) values of the corresponding words.
#    - TF-IDF is a statistical measure that reflects how important a word is to a document in a corpus.
#    - Words that appear frequently in a document but rarely across the corpus will have higher TF-IDF values.
#    - By weighting the word embeddings with TF-IDF, you can give more importance to the most relevant words in the document.
# 
# 2. **Unweighted Document Embedding**:
#    - In this representation, you need to average the word embeddings of all the words in the document without any weighting.
#    - This approach treats all words equally, regardless of their importance or frequency.
# 

# In[7]:


W2V = api.load('word2vec-google-news-300')


# In[8]:


print(W2V)


# In[9]:


vec_abs = W2V['absolute']
vec_abs


# ### Generating TF-IDF Vectors
# 
# The code is an implementation of generating the TF-IDF weighted document embeddings for the job advertisement descriptions.
# 
# 
# 
# This is a custom function `get_weighted_vector` that takes three arguments:
# - `data_features`: The TF-IDF vector representation of the documents (`tfidf_features` in your case)
# - `vocab`: The vocabulary list generated in Task 1
# - `descriptions`: The list of job advertisement descriptions
# 
# The function iterates over each job advertisement description and constructs a vector representation by combining the TF-IDF values of the words present in the description. Here's how it works:
# 
# 1. For each description, it initializes an empty list `vector_representation`.
# 2. It then iterates over each word in the vocabulary and its corresponding TF-IDF value in the current description's TF-IDF vector.
# 3. If the TF-IDF value is non-zero (i.e., the word is present in the description), it appends the TF-IDF value (as a string) to the `vector_representation` list.
# 4. After iterating over all words in the vocabulary, it prints the `vector_representation` list as an array-like string.
# 
# It imports the `TfidfVectorizer` class from scikit-learn, initializes it with the vocabulary from Task 1, and generates the TF-IDF vector representation (`tfidf_features`) for the job advertisement descriptions.
# 
# The output of this function will be a list of TF-IDF weighted document embeddings, where each embedding is represented as a vector of TF-IDF values for the words present in the corresponding job advertisement description.
# 

# In[10]:


def get_weighted_vector(data_features, vocab, descriptions):
    for d_ind, description in enumerate(descriptions):
        vector_representation = []
        for word, value in zip(vocab, data_features.toarray()[d_ind]):
            if value != 0:
                vector_representation.append(str(value))
        print("[" + ", ".join(vector_representation) + "]")


# In[11]:


from sklearn.feature_extraction.text import TfidfVectorizer
tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(descriptions) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[12]:


tfidf_features


# In[13]:


tfidf_name = tVectorizer.get_feature_names_out()


# In[14]:


get_weighted_vector(tfidf_features, vocab, descriptions)


# ### unweighted vector representation 
# 
# 
# This is for generating unweighted document embeddings using the pre-trained Word2Vec model.
# 
# 
# This is a custom function `get_average_vector` that takes two arguments:
# - `tokens`: A list of tokens (words) representing a document
# - `word2vec_model`: The pre-trained Word2Vec model
# 
# The function computes the unweighted document embedding by averaging the word embeddings of all the valid tokens (words) in the document. Here's how it works:
# 
# 1. It initializes a zero vector `vector_sum` with the same dimensionality as the word embeddings in the Word2Vec model.
# 2. It iterates over each token in the input `tokens` list.
# 3. If the token is present in the Word2Vec model's vocabulary, it adds the corresponding word embedding to the `vector_sum` and increments the `count`.
# 4. After iterating over all tokens, if the `count` is non-zero (i.e., there were valid tokens), it returns the `vector_sum` divided by the `count` (average of word embeddings).
# 5. If the `count` is zero (i.e., no valid tokens were found), it returns a zero vector with the same dimensionality as the word embeddings.
# 
# 
# Then generates the unweighted document embeddings for all the job advertisement descriptions using the `get_average_vector` function and the pre-trained Word2Vec model (`W2V`).
# 
# 1. It assumes that `tokenized_descriptions` is a list of tokenized job advertisement descriptions, where each element is a list of tokens (words).
# 2. It uses a list comprehension to apply the `get_average_vector` function to each tokenized description `desc` in `tokenized_descriptions`, passing the Word2Vec model `W2V` as the second argument.
# 3. The resulting `unweighted_vectors` is a list containing the unweighted document embeddings for all the job advertisement descriptions.
# 4. Finally, it prints the `unweighted_vectors` list.
# 

# In[15]:


from gensim.models import Word2Vec

def get_average_vector(tokens, word2vec_model):
    vector_sum = np.zeros(word2vec_model.vector_size)
    count = 0
    for token in tokens:
        if token in word2vec_model:
            vector_sum += word2vec_model[token]
            count += 1
    if count != 0:
        return vector_sum / count
    else:
        return np.zeros(word2vec_model.vector_size) 
    
unweighted_vectors = [get_average_vector(desc, W2V) for desc in tokenized_descriptions]


# In[16]:


unweighted_vectors


# In[17]:


cVectorizer = CountVectorizer(analyzer = "word",vocabulary = vocab) # initialised the CountVectorizer
cVectorizer


# ### Saving outputs
# Save the count vector representation as per spectification.
# - count_vectors.txt

# In[18]:


with open('count_vectors.txt', 'w') as out_file:
    for job in job_ads:
        webindex = job['webindex']
        description = job['description']
        count_vector = generate_count_vector(description)
        count_vector_str = ','.join(f"{index}:{count}" for index, count in count_vector.items())
        out_file.write(f"#{webindex},{count_vector_str}\n")


# ## Task 3. Job Advertisement Classification

# 
# Task 3 focuses on building machine learning models for classifying the category of job advertisement texts. The task is divided into two parts, each addressing a specific question related to the performance of different feature representations and the impact of additional information on classification accuracy.
# 
# **Q1: Language Model Comparisons**
# 
# In this part of the task, you need to investigate which language model (feature representation) generated in Task 2 performs the best with the chosen machine learning model. Specifically, you need to:
# 
# 1. Choose a machine learning model for classification (e.g., logistic regression, or any other model you prefer).
# 2. Build and evaluate the classification model using the different feature representations generated in Task 2:
#    - Count vector representation
#    - TF-IDF weighted document embedding
#    - Unweighted document embedding
# 
# 3. Compare the performance of the classification models using different feature representations to determine which language model performs the best.
# 
# **Q2: Does More Information Provide Higher Accuracy?**
# 
# This task id to explore whether incorporating additional information, such as the job title, can improve the classification accuracy. Specifically, you need to:
# 
# 1. Build and evaluate classification models using the following feature representations:
#    - Job title only
#    - Job description only (which you have already done in Q1)
#    - Both job title and job description
# 
# 2. For the feature representation combining job title and job description, you have two options:
#    - Concatenate the job title and job description text, and generate a single feature representation for the combined text.
#    - Generate separate feature representations for the job title and job description, and use both features in the classification model.
# 
# 3. Compare the performance of the classification models using different feature representations to determine if incorporating additional information (job title) improves the classification accuracy.
# 
# For both questions, you need to:
# 
# - Choose one of the language models (feature representations) that was generated in Task 2 for experimentation.
# - Perform 5-fold cross-validation to obtain robust performance evaluations and comparisons.
# 

# In[19]:


Category = [job['Category'] for job in job_ads]


# In[20]:


labels_df = pd.DataFrame({'Category': Category})
stats = labels_df['Category'].value_counts().sort_index()
print(stats)
stats.plot.bar(ylim=0)


# ###  Description of the job advertisement 

# The below code constructs and assesses the performance of a logistic regression model for categorizing job advertisements. It utilizes the TF-IDF feature representation computed in Task 2 as the input data for the model. Additionally, the code showcases the implementation of a 5-fold cross-validation technique. This approach aims to achieve a more reliable and comprehensive evaluation of the model's predictive capabilities by training and testing it on multiple subsets of the data. Cross-validation helps mitigate potential biases that may arise from using a single train-test split, thereby providing a more robust estimate of the model's generalization performance.

# In[21]:


seed = 0


# ### TF-IDF weighted 

# In[22]:


X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(tfidf_features, Category, list(range(0,len(Category))),test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# Import the necessary modules and classes from the scikit-learn library. `train_test_split` is used for splitting the data into training and test sets, `LogisticRegression` is the machine learning model for classification, and `StratifiedKFold` and `cross_val_score` are used for performing cross-validation.
# This line splits the data (`tfidf_features` and `Category`) into training and test sets. The `train_test_split` function takes the feature matrix (`tfidf_features`), target variable (`Category`), and a list of indices (to preserve the correspondence between data points and their indices). It splits the data into training and test sets, with 33% of the data used for testing (`test_size=0.33`). The `random_state` parameter ensures reproducibility of the split.
# These lines create a logistic regression model with a fixed random state, fit the model on the training data (`X_train`, `y_train`), and evaluate the model's accuracy on the test data (`X_test`, `y_test`) using the `score` method.
# 

# In[23]:


num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# Then Import the `KFold` class from scikit-learn and create a 5-fold cross-validation object (`kf`). The `KFold` class is used to generate indices for the training and validation sets in each fold. The `random_state` parameter ensures reproducibility, and `shuffle=True` ensures that the data is shuffled before splitting.
# 

# In[24]:


DW_scores = cross_val_score(model, tfidf_features, Category, cv= 5, scoring='accuracy')
DW_scores


# This performs 5-fold cross-validation on the logistic regression model using the `cross_val_score` function. It takes the model, feature matrix (`tfidf_features`), target variable (`Category`), the number of folds (`cv=5`), and the scoring metric (`scoring='accuracy'`). The function returns an array of accuracy scores for each fold, which is stored in the `scores` variable and printed.
# 
# By performing cross-validation, you can obtain a more robust estimate of the model's performance, as it evaluates the model on multiple train-test splits of the data. This helps to mitigate the potential bias introduced by a single train-test split and provides a better understanding of the model's generalization ability.

# In[25]:


tfidf_features.shape


# ### count vector

# In[26]:


count_features = cVectorizer.fit_transform(descriptions) # generate the count vector representation for all articles
count_features


# In[27]:


X_train, X_test, y_train, y_test = train_test_split(count_features, Category, test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[28]:


DCV_scores = cross_val_score(model, count_features, Category, cv= 5, scoring='accuracy')
DCV_scores


# ### unweighted vector
# 
# The code is for building and evaluating a logistic regression model for job advertisement classification using the unweighted document embeddings generated in Task 2 as the feature representation.
# 
# 
# 1. `X_train, X_test, y_train, y_test = train_test_split(unweighted_vectors, Category, test_size=0.33, random_state=seed)`
#    - I split the data (`unweighted_vectors` and `Category`) into training and test sets using the `train_test_split` function.
#    - The `unweighted_vectors` are used as the feature matrix, and `Category` is the target variable.
#    - I set `test_size=0.33` to allocate 33% of the data for testing, and the remaining 67% for training.
#    - The `random_state` parameter ensures reproducibility of the split.
# 
# 2. `model = LogisticRegression(random_state=seed)`
#    - I created an instance of the `LogisticRegression` model from scikit-learn.
#    - Setting `random_state=seed` ensures reproducibility of the model's initialization.
# 
# 3. `model.fit(X_train, y_train)`
#    - I trained the logistic regression model on the training data (`X_train`, `y_train`).
# 
# 4. `model.score(X_test, y_test)`
#    - I evaluated the trained model's accuracy on the test data (`X_test`, `y_test`) using the `score` method.
# 
# 5. `DUWscores = cross_val_score(model, unweighted_vectors, Category, cv=5, scoring='accuracy')`
#    - I performed 5-fold cross-validation on the logistic regression model using the `cross_val_score` function.
#    - The `unweighted_vectors` are used as the feature matrix, and `Category` is the target variable.
#    - I set `cv=5` to perform 5-fold cross-validation.
#    - The `scoring='accuracy'` parameter specifies that you want to evaluate the model's accuracy.
#    - The cross-validation scores are stored in the `scores` variable.
# 
# 6. `DUW_scores`
#    - I printed the cross-validation scores.
# 
# The reason I did this is to evaluate the performance of the logistic regression model for job advertisement classification using the unweighted document embeddings as the feature representation. By performing cross-validation, you can obtain a more robust estimate of the model's performance, as it evaluates the model on multiple train-test splits of the data, mitigating potential biases introduced by a single train-test split.
# 

# In[29]:


X_train, X_test, y_train, y_test = train_test_split(unweighted_vectors, Category, test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[30]:


DUW_scores = cross_val_score(model, unweighted_vectors, Category, cv= 5, scoring='accuracy')
DUW_scores


# ### Title of the job advertisement
# 
# Building and evaluating a logistic regression model for job advertisement classification using the TF-IDF feature representation of the preprocessed job titles.
# 
# 1. `titles = [job['title'] for job in job_ads]`
#    - Extracted the job titles from the `job_ads` data, which is assumed to be a list of dictionaries representing job advertisements.
# 
# 2. `import re` and `pattern = r'\b\w+\b'`
#    - Imported the `re` module for regular expressions and defined a pattern `r'\b\w+\b'` to match word tokens.
# 
# 3. `pro_titles = []` and `for title in titles: ...`
#    - Preprocessed the job titles by tokenizing them using the regular expression pattern and joining the tokens back into preprocessed titles (`pro_titles`).

# In[31]:


titles = [job['title'] for job in job_ads]


# In[32]:


pattern = r'\b\w+\b'
pro_titles = []

for title in titles:
    tokens = re.findall(pattern, title)
    pro_title = " ".join(tokens)
    pro_titles.append(pro_title)

print(pro_titles)


# ### TF-IDF weighted for Title
# 
# 4. `tVectorizer = TfidfVectorizer(analyzer="word", vocabulary=vocab)`
#    - Initialized a `TfidfVectorizer` from scikit-learn, setting the `analyzer` to "word" and using the `vocabulary` generated in Task 1.
# 
# 5. `tfidf_features = tVectorizer.fit_transform(pro_titles)`
#    - Generated the TF-IDF feature representation (`tfidf_features`) for the preprocessed job titles (`pro_titles`).
# 
# 6. `X_train, X_test, y_train, y_test, ... = train_test_split(tfidf_features, Category, ...)`
#    - Split the TF-IDF features (`tfidf_features`) and the target variable (`Category`) into training and test sets.
# 
# 7. `model = LogisticRegression(random_state=seed)`
#    - Created a logistic regression model with a fixed random state.
# 
# 8. `model.fit(X_train, y_train)` and `model.score(X_test, y_test)`
#    - Trained the logistic regression model on the training data and evaluated its accuracy on the test data.
# 
# 9. `num_folds = 5` and `kf = KFold(n_splits=num_folds, random_state=seed, shuffle=True)`
#    - Set up a 5-fold cross-validation strategy with a fixed random state and shuffling.
# 
# 10. `TW_scores = cross_val_score(model, tfidf_features, Category, cv=5, scoring='accuracy')`
#     - Performed 5-fold cross-validation on the logistic regression model using the TF-IDF features and the target variable, evaluating the model's accuracy.
# 
# The reason I did this is to investigate whether incorporating the job title information, in addition to the job description, can improve the classification accuracy for job advertisements. By generating TF-IDF feature representations for the preprocessed job titles and using them as input to the logistic regression model, I can evaluate the model's performance and compare it to the performance achieved using only the job description or a combination of job title and description.
# 
# Performing cross-validation provides a more robust estimate of the model's performance by evaluating it on multiple train-test splits of the data, mitigating potential biases introduced by a single train-test split.

# In[33]:


tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(pro_titles) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[34]:


X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(tfidf_features, Category, list(range(0,len(Category))),test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[35]:


num_folds = 5
kf = KFold(n_splits= num_folds, random_state=seed, shuffle = True) # initialise a 5 fold validation
print(kf)


# In[36]:


TW_scores = cross_val_score(model, tfidf_features, Category, cv= 5, scoring='accuracy')
TW_scores


# ### unweighted vector for Title
# 
# Building and evaluating a logistic regression model for job advertisement classification using the unweighted document embeddings of the preprocessed job titles as the feature representation.
# 
# 
# 1. `unweighted_vectors = [get_average_vector(title, W2V) for title in pro_titles]`
#    - Generated unweighted document embeddings (`unweighted_vectors`) for the preprocessed job titles (`pro_titles`) using the `get_average_vector` function and the pre-trained Word2Vec model (`W2V`).
#    - This step creates a list of unweighted document embeddings, where each embedding is the average of the word embeddings for the tokens in the corresponding job title.
# 
# 2. `X_train, X_test, y_train, y_test, ... = train_test_split(unweighted_vectors, Category, ...)`
#    - Split the unweighted document embeddings (`unweighted_vectors`) and the target variable (`Category`) into training and test sets.
# 
# 3. `model = LogisticRegression(random_state=seed)`
#    - Created a logistic regression model with a fixed random state.
# 
# 4. `model.fit(X_train, y_train)` and `model.score(X_test, y_test)`
#    - Trained the logistic regression model on the training data and evaluated its accuracy on the test data.
# 
# 5. `scores = cross_val_score(model, tfidf_features, Category, cv=5, scoring='accuracy')`
#    - Performed 5-fold cross-validation on the logistic regression model using the TF-IDF features (`tfidf_features`) of the job descriptions and the target variable (`Category`), evaluating the model's accuracy.
# 
# The reason for diong this is to investigate whether using the unweighted document embeddings of the job titles as input features can improve the classification accuracy for job advertisements, compared to using only the job descriptions or the TF-IDF feature representation of the job titles.
# 
# By generating unweighted document embeddings for the job titles using the pre-trained Word2Vec model, we captured the semantic information and context of the words in the titles. These embeddings can potentially provide a more meaningful representation of the job titles compared to the TF-IDF feature representation, which is based solely on word counts and frequencies.
# 
# You then split the unweighted document embeddings and the target variable into training and test sets, and train a logistic regression model on the training data. You evaluate the model's accuracy on the test data and perform 5-fold cross-validation using the TF-IDF features of the job descriptions.
# 
# By comparing the performance of the logistic regression model using different feature representations (TF-IDF of job titles, TF-IDF of job descriptions, unweighted document embeddings of job titles), you can investigate whether incorporating the job title information, and the type of feature representation used for the job titles, can improve the classification accuracy for job advertisements.

# In[37]:


unweighted_vectors = [get_average_vector(title, W2V) for title in pro_titles]
unweighted_vectors


# In[38]:


X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(unweighted_vectors, Category, list(range(0,len(Category))),test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[39]:


TUW_scores = cross_val_score(model, tfidf_features, Category, cv= 5, scoring='accuracy')
TUW_scores


# ###  Title and Description of the job advertisement
# 
# 
# 
# Preprocessing the job advertisement data by combining the job title and job description into a single text, and then tokenizing and cleaning the combined text.
# 
# 1. `combined_texts = []`
#    - This line initializes an empty list `combined_texts` to store the combined job title and description texts.
# 
# 2. `for job in job_ads:`
#    `    if 'title' in job and 'description' in job:`
#    `        combined_text = f"{job['title']} {job['description']}"`
#    `        combined_texts.append(combined_text)`
#    - This loop iterates over each job advertisement in the `job_ads` list.
#    - For each job advertisement, it checks if the 'title' and 'description' keys exist in the dictionary.
#    - If both keys exist, it concatenates the job title and job description into a single string `combined_text` using an f-string.
#    - The `combined_text` is then appended to the `combined_texts` list.
# 
# 3. `combined_texts`
#    - This line prints the `combined_texts` list, which now contains the combined job title and description texts for each job advertisement.
# 
# 4. `pattern = r'\b\w+\b'`
#    - This line defines a regular expression pattern `r'\b\w+\b'` to match word tokens.
# 
# 5. `combined_words = []`
#    - This line initializes an empty list `combined_words` to store the tokenized and cleaned combined texts.
# 
# 6. `for combined_text in combined_texts:`
#    `    tokens = re.findall(pattern, combined_text)`
#    `    combined_word = " ".join(tokens)`
#    `    combined_words.append(combined_word)`
#    - This loop iterates over each combined text in the `combined_texts` list.
#    - For each combined text, it uses the `re.findall` function to find all matches of the regular expression pattern `pattern` in the text.
#    - The matched tokens (words) are stored in the `tokens` list.
#    - The `tokens` list is then joined with spaces using `" ".join(tokens)` to create a cleaned and tokenized combined text `combined_word`.
#    - The cleaned and tokenized combined text `combined_word` is appended to the `combined_words` list.
# 
# 7. `print(combined_words)`
#    - This line prints the `combined_words` list, which now contains the cleaned and tokenized combined job title and description texts for each job advertisement.
# 
# this is to preprocess the job advertisement data by combining the job title and job description into a single text, and then tokenizing and cleaning the combined text. This preprocessing step is often necessary for feature extraction and modeling tasks, as it helps to clean and standardize the text data.
# 
# By combining the job title and job description, we create a single text representation that incorporates information from both sources. This combined text can then be used to generate feature representations (e.g., count vectors, word embeddings) and build classification models that leverage the information from both the job title and job description.
# 
# The tokenization and cleaning step, using the regular expression pattern `\b\w+\b`, removes any non-word characters and separates the words in the combined text. This can improve the quality of the feature representations and potentially lead to better model performance.
# 
# The cleaned and tokenized combined texts in `combined_words` can then be used for generating feature representations and building classification models that incorporate both the job title and job description information.

# In[40]:


combined_texts = []
for job in job_ads:
    if 'title' in job and 'description' in job:
        combined_text = f"{job['title']} {job['description']}"
        combined_texts.append(combined_text)


# In[41]:


pattern = r'\b\w+\b'
combined_words = []

for combined_text in combined_texts:
    tokens = re.findall(pattern, combined_text)
    combined_word = " ".join(tokens)
    combined_words.append(combined_word)



# ### TF-IDF weighted for Title and Description
# 

# In[42]:


tVectorizer = TfidfVectorizer(analyzer = "word",vocabulary = vocab) # initialised the TfidfVectorizer
tfidf_features = tVectorizer.fit_transform(combined_words) # generate the tfidf vector representation for all articles
tfidf_features.shape


# In[43]:


X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(tfidf_features, Category, list(range(0,len(Category))),test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[44]:


TDW_scores = cross_val_score(model, tfidf_features, Category, cv= 5, scoring='accuracy')
TDW_scores


# ###  unweighted for Title and Description
# 

# In[45]:


unweighted_vectors = [get_average_vector(combined_word, W2V) for combined_word in combined_words]
unweighted_vectors


# In[46]:


X_train, X_test, y_train, y_test,train_indices,test_indices = train_test_split(unweighted_vectors, Category, list(range(0,len(Category))),test_size=0.33, random_state=seed)
model = LogisticRegression(random_state=seed)
model.fit(X_train, y_train)
model.score(X_test, y_test)


# In[47]:


TDUW_scores = cross_val_score(model, unweighted_vectors, Category, cv= 5, scoring='accuracy')
TDUW_scores


# #### PLOT

# In[48]:


import matplotlib.pyplot as plt

model_names = ['Count Vector', 'TF-IDF Title', 'TF-IDF Description', 'Unweighted Title', 'Unweighted Combined']

accuracy_scores = [DCV_scores, TW_scores, DW_scores, TUW_scores, TDUW_scores]

x_pos = range(len(model_names))

fig, ax = plt.subplots(figsize=(10, 6))

ax.boxplot(accuracy_scores, labels=model_names, showmeans=True)

ax.set_title('Comparison of Model Accuracy', fontsize=16)
ax.set_xlabel('Model', fontsize=14)
ax.set_ylabel('Accuracy', fontsize=14)

plt.xticks(rotation=45)

plt.show()


# ## Summary
# 
# 
# To summarize, using the Word2Vec (GoogleNews-300) model you loaded, you need to generate the following three types of feature representations for each job advertisement description:
# 
# 1. Count Vector Representation (saved in `count_vectors.txt`)
# 2. TF-IDF Weighted Document Embedding
# 3. Unweighted Document Embedding
# 
# These feature representations capture different aspects of the text and can be useful for various natural language processing tasks, such as text classification, clustering, or information retrieval.
# 
# Task 3 involved building and evaluating machine learning models for job advertisement classification, investigating the impact of different feature representations (language models) and additional information (job title) on the classification performance.

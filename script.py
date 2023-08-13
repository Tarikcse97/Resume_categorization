#!/usr/bin/env python
# coding: utf-8

# In[3]:


import os
import re
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from keras.models import load_model


# In[6]:


# Install NLTK (if not already installed)
get_ipython().system('pip install nltk')

# Import NLTK and download stopwords
import nltk
nltk.download('stopwords')
# Load English stopwords
from nltk.corpus import stopwords
stopwords = stopwords.words('english')

from nltk.stem import WordNetLemmatizer
wordnet_lemmatizer= WordNetLemmatizer()
nltk.download('wordnet')
import re

import string
string.punctuation


# In[7]:


model_path = '~/Documents/anacondacode/archive/Resume/model.h5'
model = load_model(model_path)

vectorizer = TfidfVectorizer(max_features=5000)
wordnet_lemmatizer = WordNetLemmatizer()
label_encoder = LabelEncoder()


# In[ ]:


# def load_model_and_tokenizer(model_path):
#     inf_session = rt.InferenceSession(model_path)
#     tokenizer = AutoTokenizer.from_pretrained("distilroberta-base")
#     return inf_session, tokenizer

# def extract_text_from_pdf(pdf_path):
#     try:
#         with open(pdf_path, "rb") as pdf_file:
#             pdf_reader = PdfReader(pdf_file)
#             text = ""
#             for page_num in range(len(pdf_reader.pages)):
#                 text += pdf_reader.pages[page_num].extract_text()
#             return text
#     except Exception as e:
#         print(f"Error extracting text from PDF '{pdf_path}': {e}")
#         return ""


# In[ ]:


def categorize_resumes(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    categorized_resumes = []

    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):  # Assuming resumes are in text files
            resume_path = os.path.join(input_dir, filename)
            with open(resume_path, 'r', encoding='utf-8') as file:
                resume_text = file.read()

            preprocessed_text = preprocess_text(resume_text)
            text_vector = vectorizer.transform([preprocessed_text])

            category_idx = np.argmax(model.predict(text_vector), axis=-1)
            predicted_category = label_encoder.inverse_transform(category_idx)[0]

            category_folder = os.path.join(output_dir, predicted_category)
            if not os.path.exists(category_folder):
                os.makedirs(category_folder)

            new_path = os.path.join(category_folder, filename)
            os.rename(resume_path, new_path)

            categorized_resumes.append({'filename': filename, 'category': predicted_category})

    return categorized_resumes


# In[ ]:


input_directory = '/path/to/input/directory'
output_directory = '/path/to/output/directory'
csv_output_path = 'categorized_resumes.csv'

categorized_resumes = categorize_resumes(input_directory, output_directory)

# Save categorized_resumes to CSV
csv_data = pd.DataFrame(categorized_resumes)
csv_data.to_csv(csv_output_path, index=False)


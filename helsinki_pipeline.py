#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# # quantifying intimacy score

# # Import generic wrappers
# def calculate_intimacy_score(inputs, label=None):
#     # Define the model repo
#     model_name = "pedropei/question-intimacy" 


#     # Download pytorch model
#     model = AutoModelForSequenceClassification.from_pretrained(model_name)
#     tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)
#     # list of strings should be pretokenized
#     # add_prefix_space needed for pretokenized
    
#     t = tokenizer(inputs, return_tensors="pt", is_split_into_words=True,
#                  padding=True, truncation=True, max_length=50, add_special_tokens=True)
#     if label:
#         labels = torch.tensor(label).unsqueeze(0)
#         # Model apply
#         output = model(**t, labels=labels)
    
#     else:
#         output = model(**t)

#     return output



# inputs_fr = ["Quelle est la question que vous détestez qu'on vous pose ?", 
#              "Quelle est l'importance de l'aérodynamique dans l'espace ?", 
#              "Quelle est la meilleure façon de leur faire avoir des analsecks ?", 
#              "Qu'est-ce que l'amour à 11 ans ?"]

# labels = [0.517, -0.421, 0.4999, 0.514]

# inp_en = translation(inputs_fr)
# print(inp_en)

# score = calculate_intimacy_score(inp_en, labels)
# print(score)


# In[4]:


# language translation

from transformers import pipeline
from tqdm import tqdm
# language translation
from transformers import pipeline
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import numpy as np

from transformers import AutoModelForSequenceClassification, AutoTokenizer 
import torch



def translation(inputs_fr):
    model_checkpoint = "Helsinki-NLP/opus-mt-fr-en"
    translator = pipeline("translation", model=model_checkpoint)

    inputs = []
    for i_fr in tqdm(inputs_fr):
        inp = translator(i_fr)

        inp = inp[0]['translation_text']

        inputs.append(inp)
    
    return inputs



# In[8]:


# quantifying intimacy score
# Import generic wrappers
from scipy import stats
import torch

def predict(inputs, labels, model, tokenizer):
    # add_prefix_space needed for pretokenized
    
    predictions = []
    loss = [] if labels is not None else None
    
    for i in range(len(inputs)):
        t = tokenizer([inputs[i]], return_tensors="pt", is_split_into_words=True,
                 padding=True, truncation=True, max_length=50, add_special_tokens=True)
        if labels:
            label = torch.tensor(labels[i]).unsqueeze(0)
            output = model(**t, labels=label)
            
            loss.append(output.loss.detach().numpy())
        else:
            output = model(**t)
            
        predictions.append(torch.flatten(output.logits.data).numpy()[0])
    return predictions, loss
        
    

def get_metrics(inputs, labels, model, tokenizer):
    predictions, loss = predict(inputs, labels, model, tokenizer)
    return np.mean(loss),stats.pearsonr(predictions,labels)[0]


# In[10]:


inputs_fr = ["Quelle est la question que vous détestez qu'on vous pose ?", 
             "Quelle est l'importance de l'aérodynamique dans l'espace ?", 
             "Quelle est la meilleure façon de leur faire avoir des analsecks ?", 
             "Qu'est-ce que l'amour à 11 ans ?"]

labels = [0.517, -0.421, 0.4999, 0.514]

inp_en = translation(inputs_fr)
print(inp_en)

model_name = "pedropei/question-intimacy" 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

loss, pearsonr = get_metrics(inp_en, labels, model, tokenizer)
print('loss:',loss, 'pearsonr:',pearsonr)


# In[ ]:


import import_ipynb
from load_data import load_data, getLanguageData, k_folds, k_fold_split, shuffle_data

k=10

raw_data = load_data("data/multi_language_data/", "train_normalized.csv")['train']

#English
print('English')
en_text, en_label = getLanguageData(raw_data, 'English')

#French
print('\nFrench')
fr_text, fr_label = getLanguageData(raw_data, 'French')
fr_text_folds,fr_label_folds = k_folds(k, [fr_text], [fr_label])

#translation
text = en_text
label = en_label
print("=======================Translating French to English=============================")
for i in range(k):
    translated_text = translation(fr_text_folds[i])
    text.extend(translated_text)
    label.extend(fr_label_folds[i])
print('text:',len(text))

# label = en_label
# label.extend(fr_label)
print('Train_label:',len(label))

#Shuffle 
text, label = shuffle_data(text, label)

model_name = "pedropei/question-intimacy" 
model = AutoModelForSequenceClassification.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name, add_prefix_space=True)

print('======================HELSINKI TRANSLATION MODEL FULL PIPELINE===============')
print("Predicting Intimacy Scores")

loss, pearsonr = get_metrics(text, label, model, tokenizer)
print('loss:',loss, 'pearsonr:',pearsonr)

# score = calculate_intimacy_score(text, label)
# print(score)


# In[ ]:





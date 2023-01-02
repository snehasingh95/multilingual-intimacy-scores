# multilingual-intimacy-scores
NLP task of quantifying the intimacy-score of texts in a multi-lingual setup

## About the project
Intimacy is a fundamental aspect of how we relate to others in social settings. Language encodes the social information of intimacy through both topics and other more subtle cues (such as linguistic hedging and swearing).

We propose to predict the intimacy of tweets in a multilingual setting, as mentioned under task 9 of the 2023 SemEval challenge. The dataset consists of 9491 tweets distributed evenly in six languages - English, French, Spanish, Chinese, Portuguese, and Italian - mapped to intimacy scores in the range of 1 to 5. We use pretrained language models to translate languages to English first and then apply the huggingface/pedropei model [4] to get the intimacy scores. This baseline model provided a Pearson’s r score of 0.4796 on the English and French tweets.

Further, we explore monolingual BERT, multilingual XLM-R and XLM-T models, with down-stream training on Intimacy Score Analysis, across English, French and Chinese. We compare the models using MSE Loss and Pearson’s r metrics and observe significant improvements from the baseline models on both trained languages and zero-shot predictions. The best model shows highly positive co-relation (Pearson’s r = 0.743) between the true and predicted intimacy score, 27% above the baseline. We also present our analysis of the models and training methods of sequential vs. mixed.

## PDF Presentation
https://drive.google.com/file/d/1TqDDKOtSTWJHTXqDpThFw6FAAwb16kVs/view?usp=share_link

## Paper Report
https://drive.google.com/file/d/1njE0Iqz22sVB9x5WrkCG-9KG_LI8bvq1/view?usp=share_link

## Results
https://docs.google.com/spreadsheets/d/1lMGuU4JgN6utbl8jfcIBOZ-QolpX3VN2/edit?usp=share_link&ouid=115094911798266306032&rtpof=true&sd=true

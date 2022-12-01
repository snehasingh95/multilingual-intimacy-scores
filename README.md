# multilingual-intimacy-scores
NLP task of quantifying the intimacy-score of texts in a multi-lingual setup

## Set-up:
### Ubuntu/MacOS
#### Creating the environment

    python3 -m venv intimacy-scores
    source intimacy-scores/bin/activate

#### Installations
    python3 -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python3 -m pip install scipy tqdm transformers sentencepiece datasets import-ipynb

#### Shut-down environment
    deactivate

### Windows:
#### Creating the environment

    python3 -m venv intimacy-scores
    source intimacy-scores\Scripts\activate

#### Installations

    python -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install scipy tqdm transformers sentencepiece datasets import_ipynb
     

#### Shut-down

    deactivate



## Internal Test Evaluation:

###  hugging face -  helsinki translation + pedropei

    python3 helsinki_pipeline.py
 
## Results/Models:
Results and Models that were run are available at [multilingual-intimacy-scores/model.ipynb at huggingface · snehasingh95/multilingual-intimacy-scores (github.com)](https://github.com/snehasingh95/multilingual-intimacy-scores/blob/huggingface/model.ipynb)
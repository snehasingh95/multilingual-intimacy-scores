
# multilingual-intimacy-scores
NLP task of quantifying the intimacy-score of texts in a multi-lingual setup

## Set-up:
### Ubuntu/MacOS
#### Creating the environment

    python3 -m venv intimacy-scores
    source intimacy-scores/bin/activate

#### Installations

    #pip3 unintall torch torchvision
    python3 -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python3 -m pip install scipy tqdm transformers

#### Shut-down environment

    deactivate

### Windows:
#### Creating the environment

    python3 -m venv intimacy-scores
    source intimacy-scores\Scripts\activate

#### Installations

    #pip3 unintall torch torchvision
    python -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install scipy tqdm transformers

#### Shut-down

    deactivate

## Training:
### In English: bert-base-english

    python3 train_intimacy_model.py --mode=train \
    --model_name=bert-base-english \
    --pre_trained_model_name_or_path=bert-base-uncased \
    --train_path=data/monolingual/english_data/final_train.txt \
    --val_path=data/monolingual/english_data/final_val.txt \
    --test_path=data/monolingual/english_data/final_test.txt \
    --model_saving_path=outputs_english

### In French: bert-base-french-europeana-cased

    python3 train_intimacy_model.py --mode=train \
    --model_name=bert-base-french \
    --pre_trained_model_name_or_path=dbmdz/bert-base-french-europeana-cased \
    --train_path=data/monolingual/french_data/train.txt \
    --val_path=data/monolingual/french_data/valid.txt \
    --test_path=data/monolingual/french_data/test.txt \
    --model_saving_path=outputs_french 

### In Chinese: bert-base-chinese

    python3 train_intimacy_model.py --mode=train \
    --model_name=bert-base-chinese \
    --pre_trained_model_name_or_path=bert-base-chinese \
    --train_path=data/monolingual/chinese_data/train.txt \
    --val_path=data/monolingual/chinese_data/validate.txt \
    --test_path=data/monolingual/chinese_data/test.txt \
    --model_saving_path=outputs_chinese

## Internal Test Evaluation:

### English: bert-base-english

    python3 train_intimacy_model.py --mode=internal-test \
    --model_name=bert-base-english \
    --pre_trained_model_name_or_path=outputs_english \
    --train_path=data/monolingual/english_data/final_train.txt \
    --val_path=data/monolingual/english_data/final_val.txt \
    --test_path=data/monolingual/english_data/final_test.txt \
    --predict_data_path=data/monolingual/english_data/final_test.txt

### French: bert-base-french-europeana-cased

    python3 train_intimacy_model.py --mode=internal-test \
    --model_name=bert-base-french \
    --pre_trained_model_name_or_path=outputs_french \
    --train_path=data/monolingual/french_data/train.txt \
    --val_path=data/monolingual/french_data/valid.txt \
    --test_path=data/monolingual/french_data/test.txt \
    --predict_data_path=data/monolingual/french_data/test.txt

### Chinese: bert-base-chinese

    python3 train_intimacy_model.py --mode=internal-test \
    --model_name=bert-base-chinese \
    --pre_trained_model_name_or_path=outputs_chinese \
    --train_path=data/monolingual/chinese_data/train.txt \
    --val_path=data/monolingual/chinese_data/validate.txt \
    --test_path=data/monolingual/chinese_data/test.txt \
    --predict_data_path=data/monolingual/chinese_data/test.txt

## Results/Models:
Results and Models that were run are available at [https://drive.google.com/drive/folders/1q7UOqh8B1flNLblQWfuUXGDWfYqgAu7W?usp=sharing](https://drive.google.com/drive/folders/1q7UOqh8B1flNLblQWfuUXGDWfYqgAu7W?usp=sharing)
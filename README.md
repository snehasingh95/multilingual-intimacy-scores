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
    python3 -m pip install scipy tqdm transformers sentencepiece datasets

#### Shut-down environment

    deactivate

### Windows:
#### Creating the environment

    python3 -m venv intimacy-scores
    source intimacy-scores\Scripts\activate

#### Installations

    #pip3 unintall torch torchvision
    python -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
    python -m pip install scipy tqdm transformers sentencepiece datasets

#### Shut-down

    deactivate

## Training:
### xlm-roberta-base

Single language

    python3 train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalized.csv  --model_saving_path=outputs --lang English

Multiple languages

    python3 train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalized.csv  --model_saving_path=outputs --lang English French Chinese

Entire Dataset

    python3 train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalized.csv  --model_saving_path=outputs

## Internal Test Evaluation:

### xlm-roberta-base
Single language

    python train_intimacy_model.py --mode=internal-test --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --lang English --test_saving_path=internal_test_en.txt
    
Entire Data Set:

    python train_intimacy_model.py --mode=internal-test --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --test_saving_path=internal_test.txt 

## Inference:

### xlm-roberta-base

    python train_intimacy_model.py --mode=inference --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --test_saving_path=inference.txt


## Results/Models:
Results and Models that were run are available at [https://drive.google.com/drive/folders/12W2CSXJks7QWrV4ju2x0-el-VkS7B9P0?usp=share_link](https://drive.google.com/drive/folders/12W2CSXJks7QWrV4ju2x0-el-VkS7B9P0?usp=share_link)
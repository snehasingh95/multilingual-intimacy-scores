# multilingual-intimacy-scores
NLP task of quantifying the intimacy-score of texts in a multi-lingual setup

Set-up:

Ubuntu/MacOS
#creating the environment
python3 -m venv intimacy-scores
source intimacy-scores/bin/activate

#installations
#pip3 unintall torch torchvision
python3 -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
python3 -m pip install scipy tqdm transformers sentencepiece datasets

#shut-down
deactivate

Windows:
#creating the environment
python -m venv intimacy-scores
.\intimacy-scores\Scripts\activate

#installations
#pip3 unintall torch torchvision
python -m pip install numpy torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install scipy tqdm transformers sentencepiece datasets

#shut-down
deactivate



Training (on English only):

#xlm-roberta-base
python3 train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalized.csv  --model_saving_path=outputs --lang English

Training (on Multi-languages only): 

#xlm-roberta-base
python3 train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalized.csv  --model_saving_path=outputs --lang English French Chinese

Intimacy Score Evaluation (Internal Test Evaluation - English Only):

#xlm-roberta-base
python train_intimacy_model.py --mode=internal-test --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --lang English --test_saving_path=internal_test_en.txt

Intimacy Score Evaluation (Internal Test Evaluation - Entire Data Set):

#xlm-roberta-base
python train_intimacy_model.py --mode=internal-test --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --test_saving_path=internal_test.txt 

Testing/Inference:

#xlm-roberta-base
python train_intimacy_model.py --mode=inference --model_name=xlm-roberta-base --pre_trained_model_name_or_path=outputs --base_dir=data/multi_language_data/ --file_name=train_normalized.csv --test_saving_path=inference.txt
# multilingual-intimacy-scores
NLP task of quantifying the intimacy-score of texts in a multi-lingual setup

Set-up:

Ubuntu/MacOS
#creating the environment
python3 -m venv intimacy-scores
source intimacy-scores/bin/activate

#installations
#pip3 unintall torch torchvision
pip3 install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
pip3 install scipy tqdm transformers sentencepiece datasets

#shut-down
deactivate

Windows:
#creating the environment
python -m venv intimacy-scores
.\intimacy-scores\Scripts\activate

#installations
#pip3 unintall torch torchvision
python -m pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install scipy tqdm transformers sentencepiece datasets

#shut-down
deactivate



Training:

#roberta-base
python train_intimacy_model.py --mode=train --model_name=roberta-base --pre_trained_model_name_or_path=roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv  --model_saving_path=outputs --lang=Englishh

#xlm-roberta-base
python train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv  --model_saving_path=outputs --lang=English

Intimacy Score Evaluation (Internal Test Evaluation):

#roberta-base
python train_intimacy_model.py --mode=internal-test --model_name=roberta-base --pre_trained_model_name_or_path=roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv --lang=English --test_saving_path=ooo.txt 

#xlm-roberta-base
python train_intimacy_model.py --mode=internal-test --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv --lang=English --test_saving_path=ooo.txt 

Testing/Inference:

#roberta-base
python train_intimacy_model.py --mode=inference --model_name=roberta-base --pre_trained_model_name_or_path=roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv --test_saving_path=ooo.txt 

#xlm-roberta-base
python train_intimacy_model.py --mode=inference --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base --base_dir=data/multi_language_data/ --file_name=train_normalised.csv --test_saving_path=ooo.txt 
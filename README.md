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
pip3 install scipy tqdm transformers sentencepiece

#shut-down
deactivate

Windows:
#creating the environment
python -m venv intimacy-scores
.\intimacy-scores\Scripts\activate

#installations
#pip3 unintall torch torchvision
python -m pip install torch torchvision -f https://download.pytorch.org/whl/torch_stable.html
python -m pip install scipy tqdm transformers sentencepiece

#shut-down
deactivate



Training:

#roberta-base
python train_intimacy_model.py --mode=train --model_name=roberta-base --pre_trained_model_name_or_path=roberta-base  --train_path=data/annotated_question_intimacy_data/final_train.txt --val_path=data/annotated_question_intimacy_data/final_val.txt --test_path=data/annotated_question_intimacy_data/final_test.txt --model_saving_path=outputs

#xlm-roberta-base
python train_intimacy_model.py --mode=train --model_name=xlm-roberta-base --pre_trained_model_name_or_path=xlm-roberta-base  --train_path=data/annotated_question_intimacy_data/final_train.txt --val_path=data/annotated_question_intimacy_data/final_val.txt --test_path=data/annotated_question_intimacy_data/final_test.txt --model_saving_path=outputs



Evaluation:

# Start by finetuning distilbert to get pre-detector & conflict disambiguator
OPENAI_API_KEY=YOUR-API-KEY python -m PokeMQA-turbo_n_edited --edited-num 1 --dataset MQuAKE-CF-3k --retraining_detector --retraining_disambiguator --activate_kgprompt

# Skip the finetune stage and load the existing checkpoint of scope detector
OPENAI_API_KEY=YOUR-API-KEY python -m PokeMQA-turbo_n_edited --edited-num 3000 --dataset MQuAKE-CF-3k --detector_name detector-ckpt --dis_name dis-ckpt --activate_kgprompt

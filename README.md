# XLNetPlanCloze
This repo holds the code for the expirements described in the paper
"Facts2Story: Controlling Text Generation with Key Facts"
Presented in Coling 2020
https://www.aclweb.org/anthology/2020.coling-main.211/

The code relies heaviley on the Huggingface Transformers library. 

### To train position prediction 
run "run_xlnet_pos_predictor_improved.py" with options like so:

--output_dir=output
--model_type=posnet
--model_name_or_path=xlnet-base-cased
--do_train
--train_data_file=/home/data/train/
--num_train_epochs 10
--do_eval
--eval_data_file=/home/data/train/
--block_size 1024
--evaluate_during_training
--xlnet
--overwrite_output_dir
--logging_steps 10
--ngenres 11
--nfacts 7
--no_cuda

### To fine tune XLNet given positioned tokens
run "run_xlnet_finetuning.py" with options like so:
--output_dir=output
--model_type=xlnet
--model_name_or_path=xlnet-base-cased
--do_train
--train_data_file=/home/data/train/
--num_train_epochs
10
--do_eval
--eval_data_file=/home/data/valid/
--block_size
1024
--per_gpu_train_batch_size
1
--per_gpu_eval_batch_size
1
--evaluate_during_training
--xlnet
--overwrite_output_dir
--logging_steps
10
--single_gpu
1
--ngenres
11

### To generate stories from facts using the finetuned models
run generate_from_facts.py with appropiate arguments.

--model_type=cxlnet
--model_name_or_path=/home/models/customxlnet/checkpoint-7000
--pos_model_name_or_path=/home/models/pos_predictor/checkpoint-1000
--padding_text=""
--test_file_path=/home/data/proofed
--top_k
40
--top_p
0
--temperature
0.85
--ngenres
11
--nfacts
7





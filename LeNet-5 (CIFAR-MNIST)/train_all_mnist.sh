#------------------------BASE MODELS------------------------------
#mnist no q no batch train
python3 run.py mnist 0 -1 False True NONE ./mnist_models/mnist_net_Q0.0_no_batch.pth
#mnist no q yes batch train
python3 run.py mnist 0 -1 True True NONE ./mnist_models/mnist_net_Q0.0_yes_batch.pth

#------------------------FINETUNE TRAIN---------------------------
#mnist 0.125 q no batch finetune train
python3 run.py mnist 0.125 -1 False True ./mnist_models/mnist_net_Q0.0_no_batch.pth ./mnist_models/mnist_net_Q0.125_no_batch_finetuned.pth
#mnist 0.25 q no batch finetune train
python3 run.py mnist 0.25 -1 False True ./mnist_models/mnist_net_Q0.0_no_batch.pth ./mnist_models/mnist_net_Q0.25_no_batch_finetuned.pth
#mnist 0.5 q no batch finetune train
python3 run.py mnist 0.5 -1 False True ./mnist_models/mnist_net_Q0.0_no_batch.pth ./mnist_models/mnist_net_Q0.5_no_batch_finetuned.pth

#mnist 0.125 q yes batch finetune train
python3 run.py mnist 0.125 -1 True True ./mnist_models/mnist_net_Q0.0_yes_batch.pth ./mnist_models/mnist_net_Q0.125_yes_batch_finetuned.pth
#mnist 0.25 q yes batch finetune train
python3 run.py mnist 0.25 -1 True True ./mnist_models/mnist_net_Q0.0_yes_batch.pth ./mnist_models/mnist_net_Q0.25_yes_batch_finetuned.pth
#mnist 0.5 q yes batch finetune train
python3 run.py mnist 0.5 -1 True True ./mnist_models/mnist_net_Q0.0_yes_batch.pth ./mnist_models/mnist_net_Q0.5_yes_batch_finetuned.pth

#-----------------------QAT TRAIN---------------------------------
#QAT round yes weight yes 
python3 run.py mnist 0.125 -1 True True NONE ./mnist_models/mnist_net_Q0.125_batch_qat_yes_round_yes_weight2.pth True True
#QAT round no weight yes
python3 run.py mnist 0.125 -1 True True NONE ./mnist_models/mnist_net_Q0.125_batch_qat_no_round_yes_weight2.pth False True
#QAT round yes weight no
python3 run.py mnist 0.125 -1 True True NONE ./mnist_models/mnist_net_Q0.125_batch_qat_yes_round_no_weight2.pth True False
#QAT round no weight no
python3 run.py mnist 0.125 -1 True True NONE ./mnist_models/mnist_net_Q0.125_batch_qat_no_round_no_weight2.pth False False

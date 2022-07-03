#------------------------BASE MODELS------------------------------
#cifar no q no batch train
python3 run.py cifar 0 -1 False True NONE ./cifar_models/cifar_net_Q0.0_no_batch.pth
#cifar no q yes batch train
python3 run.py cifar 0 -1 True True NONE ./cifar_models/cifar_net_Q0.0_yes_batch.pth

#------------------------FINETUNE TRAIN---------------------------
#cifar 0.125 q no batch finetune train
python3 run.py cifar 0.125 -1 False True ./cifar_models/cifar_net_Q0.0_no_batch.pth ./cifar_models/cifar_net_Q0.125_no_batch_finetuned.pth
#cifar 0.25 q no batch finetune train
python3 run.py cifar 0.25 -1 False True ./cifar_models/cifar_net_Q0.0_no_batch.pth ./cifar_models/cifar_net_Q0.25_no_batch_finetuned.pth
#cifar 0.5 q no batch finetune train
python3 run.py cifar 0.5 -1 False True ./cifar_models/cifar_net_Q0.0_no_batch.pth ./cifar_models/cifar_net_Q0.5_no_batch_finetuned.pth

#cifar 0.125 q yes batch finetune train
python3 run.py cifar 0.125 -1 True True ./cifar_models/cifar_net_Q0.0_yes_batch.pth ./cifar_models/cifar_net_Q0.125_yes_batch_finetuned.pth
#cifar 0.25 q yes batch finetune train
python3 run.py cifar 0.25 -1 True True ./cifar_models/cifar_net_Q0.0_yes_batch.pth ./cifar_models/cifar_net_Q0.25_yes_batch_finetuned.pth
#cifar 0.5 q yes batch finetune train
python3 run.py cifar 0.5 -1 True True ./cifar_models/cifar_net_Q0.0_yes_batch.pth ./cifar_models/cifar_net_Q0.5_yes_batch_finetuned.pth

#-----------------------QAT TRAIN---------------------------------
#QAT round yes weight yes 
python3 run.py cifar 0.125 -1 True True NONE ./cifar_models/cifar_net_Q0.125_batch_qat_yes_round_yes_weight2.pth True True
#QAT round no weight yes
python3 run.py cifar 0.125 -1 True True NONE ./cifar_models/cifar_net_Q0.125_batch_qat_no_round_yes_weight2.pth False True
#QAT round yes weight no
python3 run.py cifar 0.125 -1 True True NONE ./cifar_models/cifar_net_Q0.125_batch_qat_yes_round_no_weight2.pth True False
#QAT round no weight no
python3 run.py cifar 0.125 -1 True True NONE ./cifar_models/cifar_net_Q0.125_batch_qat_no_round_no_weight2.pth False False

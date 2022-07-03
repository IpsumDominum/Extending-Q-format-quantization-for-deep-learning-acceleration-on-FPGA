#cifar no q no batch test
python3 run.py cifar 0 -1 False False ./cifar_models/cifar_net_Q0.0_no_batch.pth "cifar no q no batch test"
#cifar 0.125 q no batch test
python3 run.py cifar 0.125 9 False False ./cifar_models/cifar_net_Q0.0_no_batch.pth "cifar 0.125 q no batch test"
#cifar 0.25 q no batch test
python3 run.py cifar 0.25 2 False False ./cifar_models/cifar_net_Q0.0_no_batch.pth "cifar 0.25 q no batch test"
#cifar 0.5 q no batch test
python3 run.py cifar 0.5 3 False False ./cifar_models/cifar_net_Q0.0_no_batch.pth "cifar 0.5 q no batch test"

#cifar no q yes batch test
python3 run.py cifar 0 -1 True False ./cifar_models/cifar_net_Q0.0_yes_batch.pth "cifar no q yes batch test"
#cifar 0.125 q yes batch test
python3 run.py cifar 0.125 1 True False ./cifar_models/cifar_net_Q0.0_yes_batch.pth "cifar 0.125 q yes batch test"
#cifar 0.25 q yes batch test
python3 run.py cifar 0.25 2 True False ./cifar_models/cifar_net_Q0.0_yes_batch.pth "cifar 0.25 q yes batch test"
#cifar 0.5 q yes batch test
python3 run.py cifar 0.5 3 True False ./cifar_models/cifar_net_Q0.0_yes_batch.pth "cifar 0.5 q yes batch test"

#------------------------FINETUNE TEST----------------------------
#cifar 0.125 q no batch finetune test
python3 run.py cifar 0.125 -1 False False ./cifar_models/cifar_net_Q0.125_no_batch_finetuned.pth "cifar 0.125 q no batch finetune test"
#cifar 0.125 q no batch finetune test
python3 run.py cifar 0.25 -1 False False ./cifar_models/cifar_net_Q0.25_no_batch_finetuned.pth "cifar 0.25 q no batch finetune test"
#cifar 0.125 q no batch finetune test
python3 run.py cifar 0.5 -1 False False ./cifar_models/cifar_net_Q0.5_no_batch_finetuned.pth "cifar 0.5 q no batch finetune test"

#cifar 0.125 q yes batch finetune test
python3 run.py cifar 0.125 -1 True False ./cifar_models/cifar_net_Q0.125_yes_batch_finetuned.pth "cifar 0.125 q yes batch finetune test"
#cifar 0.125 q yes batch finetune test
python3 run.py cifar 0.25 -1 True False ./cifar_models/cifar_net_Q0.25_yes_batch_finetuned.pth "cifar 0.25 q yes batch finetune test"
#cifar 0.125 q yes batch finetune test
python3 run.py cifar 0.5 -1 True False ./cifar_models/cifar_net_Q0.5_yes_batch_finetuned.pth "cifar 0.5 q yes batch finetune test"

#------------------------QAT TEST---------------------------------
#QAT round yes weight yes 
python3 run.py cifar 0.125 -1 True False ./cifar_models/cifar_net_Q0.125_batch_qat_yes_round_yes_weight.pth "QAT round yes weight yes "
#QAT round no weight yes
python3 run.py cifar 0.125 -1 True False ./cifar_models/cifar_net_Q0.125_batch_qat_no_round_yes_weight.pth "QAT round no weight yes"
#QAT round yes weight no
python3 run.py cifar 0.125 -1 True False ./cifar_models/cifar_net_Q0.125_batch_qat_yes_round_no_weight.pth "QAT round yes weight no"
#QAT round no weight no
python3 run.py cifar 0.125 -1 True False ./cifar_models/cifar_net_Q0.125_batch_qat_no_round_no_weight.pth "QAT round no weight no"
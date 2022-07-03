import torch
import matplotlib.pyplot as plt
import os
dataset = "cifar"
root = dataset+"_models"
match_name = ["S+W","S Only","W Only","No S No W","Baseline No Q"]
match_name = ["0.125","0.25","0.5","Baseline No Q"]
names = [
    #dataset+"_net_Q0.125_batch_qat_yes_round_yes_weight.pth",
    #dataset+"_net_Q0.125_batch_qat_yes_round_no_weight.pth",
    #dataset+"_net_Q0.125_batch_qat_no_round_yes_weight.pth",
    #dataset+"_net_Q0.125_batch_qat_no_round_no_weight.pth",
    dataset+"_net_Q0.125_yes_batch_finetuned.pth",
    dataset+"_net_Q0.25_yes_batch_finetuned.pth",
    dataset+"_net_Q0.5_yes_batch_finetuned.pth",
    dataset+"_net_Q0.0_yes_batch.pth",
]
for jdx,item in enumerate(names):
    PATH = os.path.join(root,item)
    checkpoint = torch.load(PATH)
    a = checkpoint["test_accuracies"]
    plt.plot(a,label=match_name[jdx])
    print(jdx,item)
plt.title(dataset.upper()+" FINETUNE")
plt.legend(loc="upper left")
plt.show()
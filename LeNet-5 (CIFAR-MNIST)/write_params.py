import torch
from model import Lenet
import os

dataset = "mnist"
SHAPE = 800
CHANNELS = 1
ROUND_NUMBER = 0.125
net = Lenet(SHAPE,True,CHANNELS,affine=False)
root = dataset+"_models"
PATH = dataset+"_net_Q0.125_batch.pth"
checkpoint = torch.load(os.path.join(root,PATH))
net.load_state_dict(checkpoint['model_state_dict'])
save_dir = os.path.join("saved_weights",PATH)
if(not os.path.isdir(save_dir)):
    os.makedirs(save_dir)

idx = 1
jdx = 1
for name,data in net.named_parameters():
    if("weight" in name):
        save_name = "weight"+str(idx)
        idx +=1
    elif("bias" in name):
        save_name = "bias"+str(jdx)
        jdx +=1
    else:
        continue
        data = torch.round(data/ROUND_NUMBER)*ROUND_NUMBER
        save_name = name.replace(".","_")
    with open(os.path.join(save_dir,save_name+".txt"),"w") as file:
        for item in data.flatten():
            file.write(str(item.item())+"\n")


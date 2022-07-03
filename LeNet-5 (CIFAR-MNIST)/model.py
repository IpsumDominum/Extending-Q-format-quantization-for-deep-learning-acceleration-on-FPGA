import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import os
random_seed = 0
torch.manual_seed(random_seed)
from sys import argv

def get_bool(arg):
    if(arg.lower()=="true"):
        return True
    else:
        return False
try:
    GPU = True
    DATASET = argv[1].lower()
    Q_NUM = float(argv[2])
    CAP_DIM = float(argv[3])
    ROUND_NUMBER = Q_NUM
    USE_BATCH_NORM = get_bool(argv[4])
    TRAIN = get_bool(argv[5])
    if(argv[5].lower()=="true"):
        TRAIN = True
    else:
        TRAIN = False
    LOAD_PATH = str(argv[6])
    SAVE_PATH = str(argv[7])
    print(LOAD_PATH)
    FIXED = False    
    try:
        ROUND = get_bool(argv[8])
    except IndexError:
        ROUND = False
    try:
        WEIGHT_INIT = get_bool(argv[9])
    except IndexError:
        WEIGHT_INIT = True
    if(LOAD_PATH.upper()=="NONE"):
        LOAD_PATH = ""
    if(SAVE_PATH.upper()=="NONE"):
        SAVE_PATH = f"{DATASET}_undefined"
    if(USE_BATCH_NORM):
        id_string = f"{DATASET}_net_Q{str(Q_NUM)}_batch"
    else:
        id_string = f"{DATASET}_net_Q{str(Q_NUM)}_no_batch"
    if(Q_NUM!=0):
        USE_Q = True
    else:
        USE_Q = False
    assert DATASET=="mnist" or DATASET=="cifar"
    if(GPU==False):
        device = "cpu"
    else:
        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
except IndexError:
    print("BAD INDEX")
    exit()

print(f"TRAIN: {TRAIN} | Q_NUM: {Q_NUM} | USE_Q: {USE_Q} | USE BNORM: {USE_BATCH_NORM}")
print(f"LOAD: {LOAD_PATH}")
print(f"SAVE: {SAVE_PATH}")
print(f"WEIGHT INIT: {WEIGHT_INIT} ROUND: {ROUND}")
if(LOAD_PATH!="" and Q_NUM!=0):
    print("FINETUNING...")
if(LOAD_PATH=="" and Q_NUM!=0):
    print("QAT...")
if(os.path.isfile(SAVE_PATH)):
    print("ALREADY DONE...")
    exit()
if(TRAIN==True):
    VIEW_DIM = False
else:
    VIEW_DIM = True
VIEW_DIM_ORIGINAL = VIEW_DIM
CAP_DIM = -1
MAX_DIM = 0
MIN_DIM = 10000000
AVG_DIM = []
VAR = {}
MEAN = {}
from collections import defaultdict
import math
VAR_FOUND = defaultdict(lambda:False)
import numpy as np
def get_max(item):
    max_val = torch.max(torch.abs(item.flatten()))
    try:
        try:            
            dim = math.ceil(math.log2(max_val))
        except OverflowError:            
            dim = 0
        if(max_val>(2**(dim)-ROUND_NUMBER)):
            dim += 1
    except ValueError:
        return 0
    if(dim<0):
        if(max_val<(1-ROUND_NUMBER)):
            dim = 0
        else:
            dim = 1
    return dim

"""
Batch norm code modifed from :
http://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=QuantizeBatchNorm
"""
num = 0
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum,affine=False,train=True):
    global VAR,MEAN,VAR_FOUND,MIN_DIM,MAX_DIM,AVG_DIM,num
    if not train:                
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            val = X - mean
            var = (val ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            val = X - mean
            var = (val ** 2).mean(dim=(0, 2, 3), keepdim=True)        
        if(VIEW_DIM):
            v = get_max(var)
            m = get_max(mean)
            if(v<MIN_DIM):
                MIN_DIM = v
            if(v>MAX_DIM):
                MAX_DIM = v
            if(m<MIN_DIM):
                MIN_DIM = m
            if(m>MAX_DIM):
                MAX_DIM = m
            AVG_DIM.append(v)
            AVG_DIM.append(m)
            #print("VAR",v)
            #print("MEAN",m)
        if(VAR_FOUND[X.shape[1]]==False):
            VAR[X.shape[1]] = var
            MEAN[X.shape[1]] = mean
            VAR_FOUND[X.shape[1]] = True            
            #X_hat = gamma*((X - mean) / torch.sqrt(var + eps)) + beta
            #neg_mean = mean-beta/(gamma/ torch.sqrt(var + eps))
            #inv_var = gamma/torch.sqrt(var + eps)
            #neg_mean = Quantize.apply(neg_mean)
            #inv_var = Quantize.apply(inv_var)
            #X_hat = (X - neg_mean) * (inv_var)
            var = Quantize.apply(var)
            mean = Quantize.apply(mean)
            X_hat = (X-mean)/torch.sqrt(var+eps)
            #mean = Quantize.apply(mean)
            #var = Quantize.apply(var)
            #X_hat = (X - mean) / torch.sqrt(var + eps)
            """
            with open(os.path.join("saved_weights",LOAD_PATH.replace("./","").replace("mnist_models/",""),f"batch{num+1}_inv_sd.txt"),"w") as file:
                for val in inv_var.flatten():
                    file.write(str(val.item())+"\n")
            with open(os.path.join("saved_weights",LOAD_PATH.replace("./","").replace("mnist_models/",""),f"batch{num+1}_neg_mean.txt"),"w") as file:
                for val in neg_mean.flatten():
                    file.write(str(-val.item())+"\n")
            """
            num = (num+1) % 4
        else:
            #moving_var = Quantize.apply(moving_var)
            #moving_mean = Quantize.apply(moving_mean)
            moving_mean = MEAN[X.shape[1]]
            moving_var = VAR[X.shape[1]]
            #X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
            #neg_mean = moving_mean-beta/(gamma/ torch.sqrt(moving_var + eps))
            #inv_var = gamma/ torch.sqrt(moving_var + eps)
            #neg_mean = Quantize.apply(neg_mean)
            #inv_var = Quantize.apply(inv_var)
            #X_hat = (X - neg_mean) * (inv_var)
            moving_var = Quantize.apply(moving_var)
            moving_mean = Quantize.apply(moving_mean)            
            X_hat = (X - moving_mean)/torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        #neg_mean = mean-beta/(gamma/ torch.sqrt(var + eps))
        #inv_var = gamma/ torch.sqrt(var + eps)
        #neg_mean = Quantize.apply(neg_mean)
        #inv_var = Quantize.apply(inv_var)
        #X_hat = (X - neg_mean) * (inv_var)
        #X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        #X_hat = X_hat*gamma + beta
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    gamma = Quantize.apply(gamma)
    beta = Quantize.apply(beta)
    Y = X_hat*gamma + beta
    #Y = X_hat
    return Y, moving_mean.data, moving_var.data
class QuantizeBatchNorm(nn.Module):
    # `num_features`: the number of outputs for a fully-connected layer
    # or the number of output channels for a convolutional layer. `num_dims`:
    # 2 for a fully-connected layer and 4 for a convolutional layer
    def __init__(self, num_features, num_dims,affine=False,train=True):
        super().__init__()
        if num_dims == 2:
            shape = (1, num_features)
        else:
            shape = (1, num_features, 1, 1)
        # The scale parameter and the shift parameter (model parameters) are
        # initialized to 1 and 0, respectively        
        affine = True
        self.affine = affine
        self.train_or_not = TRAIN
        if(affine):
            self.gamma = nn.Parameter(torch.ones(shape))
            self.beta = nn.Parameter(torch.zeros(shape))
        else:
            self.gamma = None
            self.beta = None            
        # The variables that are not model parameters are initialized to 0 and 1
        self.moving_mean = torch.zeros(shape)
        self.moving_var = torch.ones(shape)
    def forward(self, X):
        # If `X` is not on the main memory, copy `moving_mean` and
        # `moving_var` to the device where `X` is located
        if self.moving_mean.device != X.device:
            self.moving_mean = self.moving_mean.to(X.device)
            self.moving_var = self.moving_var.to(X.device)
        # Save the updated `moving_mean` and `moving_var`
        Y, self.moving_mean, self.moving_var = batch_norm(
            X, self.gamma, self.beta, self.moving_mean,
            self.moving_var, eps=1e-5, momentum=0.1,affine=self.affine,train=self.train_or_not)
        return Y

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if(USE_Q):
            #ctx.save_for_backward(x)
            if(CAP_DIM!=-1):
                min_val = -2**(CAP_DIM+1)+ROUND_NUMBER
                max_val = 2**(CAP_DIM+1)-ROUND_NUMBER
                #print(min_val,max_val)
                rounded = torch.round(x/ROUND_NUMBER)*ROUND_NUMBER
                rounded = torch.clamp(rounded,min_val,max_val)
            else:
                rounded = torch.round(x/ROUND_NUMBER)*ROUND_NUMBER
            #print(torch.mean(torch.abs(rounded-x)))
            return rounded
        else:
            return x
    @staticmethod
    def backward(ctx, grad_output):
        #if(USE_Q):
        #    x, = ctx.saved_tensors
        return grad_output
        #else:
        #    return grad_output

class Lenet(nn.Module):
    def __init__(self,intermediate_node=1250,train=True,channels=3,affine=False):
        super(Lenet, self).__init__()
        #input channel 1, output channel 10
        
        self.conv1 = nn.Conv2d(channels, 20, kernel_size=5, stride=1)
        self.bn1 = QuantizeBatchNorm(20,4,affine=affine,train=train) 
        self.mp1 = nn.MaxPool2d(2,None)
        self.conv2 = nn.Conv2d(20, 50, kernel_size=5, stride=1)
        self.bn2 = QuantizeBatchNorm(50,4,affine=affine,train=train)
        self.mp2 = nn.MaxPool2d(2,None)
        self.flat = nn.Flatten()        
        self.l1 = nn.Linear(intermediate_node,500)
        self.bn3 = QuantizeBatchNorm(500,2,affine=affine,train=train)
        self.l2 = nn.Linear(500,10)
        self.bn4 = QuantizeBatchNorm(10,2,affine=affine,train=train)
        self.q = Quantize.apply
        self.train_or_not = TRAIN
        self.affine = affine        
        #if(train):
        #if(USE_BATCH_NORM):
        if(USE_Q):
            self.apply(self._init_weights)
        #if(USE_Q):
        #    self.apply(self.quantize_weights)
    def _init_weights(self, module):        
        if isinstance(module, nn.Linear):                        
            if(WEIGHT_INIT==False):
                module.weight.data.normal_(mean=0.0, std=ROUND_NUMBER)
            else:
                module.weight.data.normal_(mean=0.0, std=1)
            if module.bias is not None:
                module.bias.data.fill_(ROUND_NUMBER)
        elif isinstance(module, nn.Conv2d):
            if(WEIGHT_INIT==False):
                module.weight.data.normal_(mean=0.0, std=ROUND_NUMBER)
            else:
                module.weight.data.normal_(mean=0.0, std=1)
            #module.weight.data.normal_(mean=0.0, std=ROUND_NUMBER)
            if module.bias is not None:
                module.bias.data.fill_(ROUND_NUMBER)
    def quantize_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data = self.q(module.weight.data)
            module.bias.data = self.q(module.bias.data)
        elif isinstance(module, QuantizeBatchNorm):
            module.moving_var = self.q(module.moving_var)
            module.moving_mean = self.q(module.moving_mean)
        elif isinstance(module, nn.Conv2d):
            module.weight.data = self.q(module.weight.data)
            module.bias.data = self.q(module.bias.data)
    def forward(self, x):                
        layer_outs = []
        x = self.q(x)
        x = self.conv1(x)
        if(TRAIN==False):
            layer_outs.append(x)
        x = self.q(x)
        if(USE_BATCH_NORM):
            x = self.bn1(x)
            if(TRAIN==False):
                layer_outs.append(x)
            x = self.q(x)
        x = self.mp1(x)
        x = F.relu(x)
        x = self.q(x)
        x = self.conv2(x)
        if(TRAIN==False):
            layer_outs.append(x)
        x = self.q(x)
        if(USE_BATCH_NORM):
            x = self.bn2(x)
            if(TRAIN==False):
                layer_outs.append(x)
            x = self.q(x)
        x = self.mp2(x)
        x = F.relu(x)
        x = self.q(x)
        x = self.flat(x)
        x = self.l1(x)
        if(TRAIN==False):
            layer_outs.append(x)
        x = self.q(x)
        if(USE_BATCH_NORM):
            x = self.bn3(x)
            if(TRAIN==False):
                layer_outs.append(x)
            x = self.q(x)
        x = F.relu(x)
        x = self.q(x)
        x = self.l2(x)
        if(TRAIN==False):
            layer_outs.append(x)
        x = self.q(x)
        if(USE_BATCH_NORM):
            x = self.bn4(x)
            if(TRAIN==False):
                layer_outs.append(x)
            x = self.q(x)
        if(USE_Q):
            global VIEW_DIM
            VIEW_DIM = False
            self.apply(self.quantize_weights)
            VIEW_DIM = VIEW_DIM_ORIGINAL
        if(TRAIN==False):
            return F.softmax(x),layer_outs
        else:
            return F.softmax(x)

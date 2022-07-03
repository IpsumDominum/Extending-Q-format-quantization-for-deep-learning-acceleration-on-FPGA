import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
random_seed = 0
torch.manual_seed(random_seed)

Q_NUM = 0.125
ROUND_NUMBER = 0.125
USE_Q = True
USE_BATCH_NORM = True
FIXED = False
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
if(USE_Q):
    id_string = str(Q_NUM)
else:
    id_string = "No_Q"
if(USE_BATCH_NORM):
    id_string += "_Batch"

"""
Batch norm code modifed from :
http://d2l.ai/chapter_convolutional-modern/batch-norm.html?highlight=QuantizeBatchNorm
"""
def batch_norm(X, gamma, beta, moving_mean, moving_var, eps, momentum,affine=False,train=True):
    if not train:        
        X_hat = (X - moving_mean) / torch.sqrt(moving_var + eps)
    else:
        assert len(X.shape) in (2, 4)
        if len(X.shape) == 2:
            mean = X.mean(dim=0)
            var = ((X - mean) ** 2).mean(dim=0)
        else:
            mean = X.mean(dim=(0, 2, 3), keepdim=True)
            var = ((X - mean) ** 2).mean(dim=(0, 2, 3), keepdim=True)
        X_hat = (X - mean) / torch.sqrt(var + eps)
        moving_mean = momentum * moving_mean + (1.0 - momentum) * mean
        moving_var = momentum * moving_var + (1.0 - momentum) * var
    if(affine):
        Y = gamma * X_hat + beta  # Scale and shift
    else:
        Y = X_hat
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
        self.affine = affine
        self.train_or_not = train
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
            self.moving_var, eps=1e-5, momentum=0.9,affine=self.affine,train=self.train_or_not)
        return Y

class Quantize(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x):
        if(USE_Q):
            ctx.save_for_backward(x)
            return torch.round(x/ROUND_NUMBER)*ROUND_NUMBER
        else:
            return x
    @staticmethod
    def backward(ctx, grad_output):
        if(USE_Q):
            x, = ctx.saved_tensors
            return grad_output*x-torch.round(x/ROUND_NUMBER)*ROUND_NUMBER
        else:
            return grad_output

class Lenet(nn.Module):
    def __init__(self,intermediate_node=1250,train=True,channels=3,affine=False):
        super(Lenet, self).__init__()
        #input channel 1, output channel 10
        self.model = nn.Sequential(
            nn.Conv2d(channels, 20, kernel_size=5, stride=1),            
            #nn.BatchNorm2d(20,affine=affine),
            nn.MaxPool2d(2,None),
            nn.ReLU(),
            nn.Conv2d(20, 50, kernel_size=5, stride=1),            
            #nn.BatchNorm2d(50,affine=affine),
            nn.MaxPool2d(2,None),
            nn.ReLU(),
            nn.Flatten(),       
            nn.Linear(intermediate_node,500),            
            #nn.BatchNorm1d(500,affine=affine),
            nn.ReLU(),
            nn.Linear(500,10),
            #nn.BatchNorm1d(10,affine=affine),
            nn.Softmax()
        )
        self.model = nn.Sequential(torch.quantization.QuantStub(), 
                  *self.model, 
                  torch.quantization.DeQuantStub())
        if(train):
            backend = "fbgemm"  # running on a x86 CPU. Use "qnnpack" if running on ARM.
            self.model.train()
            self.model.qconfig = torch.quantization.get_default_qconfig(backend)
            torch.quantization.prepare_qat(self.model, inplace=True)
    def forward(self, x):                
        return self.model(x)

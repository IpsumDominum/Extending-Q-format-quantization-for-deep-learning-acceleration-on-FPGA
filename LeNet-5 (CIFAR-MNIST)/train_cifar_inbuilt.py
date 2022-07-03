import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
from model_inbuilt import Lenet,id_string,ROUND_NUMBER,USE_Q,USE_BATCH_NORM,Q_NUM,FIXED
random_seed = 0
torch.manual_seed(random_seed)

transform = transforms.Compose(
    [transforms.ToTensor(),])
#transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

batch_size = 32

trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=2048,
                                         shuffle=True, num_workers=2)
classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
net = Lenet(1250,True,affine=False)
net.to(device)
import torch.optim as optim
from tqdm import tqdm
#criterion = nn.CrossEntropyLoss().to(device)
criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
#scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.1, steps_per_epoch=len(trainloader), epochs=max_epoches)
test_losses = []
training_losses = []
train_accuracies = []
test_accuracies = []

PATH = ""
#PATH = "cifar_net_no_q_batch.pth"
#PATH = "cifar_net_no_q_no_batch.pth"
#PATH = "cifar_net_no_q_batch_affine_fixed.pth"
if(PATH!=""):
    checkpoint = torch.load("cifar_models_trained/"+PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint["train_losses"]
    test_losses = checkpoint["test_losses"]
    train_accuracies= checkpoint["train_accuracies"]
    test_accuracies = checkpoint["test_accuracies"]
    net.train()
    max_epoches = 5
else:
    max_epoches = 50

for epoch in range(max_epoches):  # loop over the dataset multiple times
    running_loss = 0.0    
    running_test_loss = 0.0
    amount_train = 0
    amount_test = 0
    #if(epoch>5):
    #    USE_Q = True    
    correct_count = 0
    total_count = 0
    i = 0
    for data in tqdm(trainloader):
        i+=1
        # get the inputs; data is a list of [inputs, labels]        
        ROUND_NUMBER = (Q_NUM/(len(trainloader)*(max_epoches)))*(len(trainloader)*epoch+i)+0.001
        inputs, labels = data
        inputs = inputs.type(torch.cuda.FloatTensor)
        labels = labels.type(torch.cuda.LongTensor)
        # zero the parameter gradients
        optimizer.zero_grad()
        # forward + backward + optimize
        #outputs = net(inputs)
        """
        USE_Q = True
        outputs,quantize_outs = net(inputs)
        USE_Q = False
        outputs,normal_outs = net(inputs)
        #mean_error = 0
        for i in range(len(quantize_outs)):            
            mean_error += torch.mean(torch.pow(quantize_outs[i]-normal_outs[i],2))
        labels_new = F.one_hot(labels,num_classes=len(classes)).type(torch.cuda.FloatTensor)
        loss = criterion(outputs, labels_new)+mean_error/3200
        """
        outputs = net(inputs)
        labels_new = F.one_hot(labels,num_classes=len(classes)).type(torch.cuda.FloatTensor)
        loss = criterion(outputs, labels_new)
        loss.backward()
        optimizer.step()
        #scheduler.step()
        # print statistics
        running_loss += loss.item()        
        amount_train += 1
        out_argmax = torch.argmax(outputs,axis=1)
        for idx,item in enumerate(out_argmax):
            if(labels[idx]==item):
                correct_count+=1
            total_count+=1    
    train_accuracy = correct_count/total_count
    correct_count = 0
    total_count = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]        
            inputs, labels = data
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients        
            # forward + backward + optimize
            outputs = net(inputs)
            labels_new = F.one_hot(labels,num_classes=len(classes)).type(torch.cuda.FloatTensor)
            loss = criterion(outputs, labels_new)
            # print statistics
            running_test_loss += loss.item()
            amount_test += 1
            out_argmax = torch.argmax(outputs,axis=1)
            for idx,item in enumerate(out_argmax):
                if(labels[idx]==item):
                    correct_count+=1
                total_count+=1
    test_accuracy = correct_count/total_count
    print(f'{id_string} [{epoch + 1},] loss: {running_loss/amount_train:.3f} test_loss: {running_test_loss/amount_test:.3f} Round Number: {ROUND_NUMBER} Train Acc: {train_accuracy} Test Acc: {test_accuracy}')
    training_losses.append(running_loss/amount_train)
    test_losses.append(running_test_loss/amount_test)
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    

print('Finished Training')
"""
if(PATH==""):
    if(USE_Q==False):
        if(USE_BATCH_NORM):
            PATH = './cifar_models_trained/cifar_net_no_q_batch_round.pth'
        else:
            PATH = './cifar_models_trained/cifar_net_no_q_no_batch.pth'
    else:
        if(USE_BATCH_NORM):
            PATH = './cifar_models_trained/cifar_net_'+str(Q_NUM)+'_batch.pth'
        else:
            PATH = './cifar_models_trained/cifar_net_'+str(Q_NUM)+'_no_batch.pth'
else:
    PATH = "./cifar_models/"+PATH
"""
PATH = "inbuilt_quantize_cifar_no_batch.pth"
torch.save(
    {
    "model_state_dict":net.state_dict(),
    "optimizer_state_dict":optimizer.state_dict(),
    "train_losses":training_losses,
    "test_losses":test_losses,
    "train_accuracies":train_accuracies,
    "test_accuracies":test_accuracies},PATH)


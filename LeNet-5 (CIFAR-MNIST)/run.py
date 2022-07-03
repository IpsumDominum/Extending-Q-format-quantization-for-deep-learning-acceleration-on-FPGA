import torch
import torchvision
from model import Lenet,id_string,ROUND_NUMBER,USE_Q,USE_BATCH_NORM,Q_NUM,FIXED,ROUND,LOAD_PATH,SAVE_PATH,TRAIN,DATASET,GPU,VIEW_DIM,get_max,MAX_DIM,MIN_DIM,AVG_DIM
import numpy as np
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F

batch_size_train = 32 # We use a small batch size here for training
batch_size_test = 2048 #
transform = transforms.Compose(
    [transforms.ToTensor(),])
if(DATASET=="cifar"):
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size_train,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size_test,
                                            shuffle=True, num_workers=2)
    SHAPE = 1250
    CHANNELS = 3
else:
    # define how image transformed
    image_transform = torchvision.transforms.Compose([
                                torchvision.transforms.ToTensor()])
    #image datasets
    train_dataset = torchvision.datasets.MNIST('dataset/', 
                                            train=True, 
                                            download=True,
                                            transform=image_transform)
    test_dataset = torchvision.datasets.MNIST('dataset/', 
                                            train=False, 
                                            download=True,
                                            transform=image_transform)
    #data loaders
    trainloader = torch.utils.data.DataLoader(train_dataset,
                                            batch_size=batch_size_train, 
                                            shuffle=True)
    testloader = torch.utils.data.DataLoader(test_dataset,
                                            batch_size=batch_size_test, 
                                            shuffle=True)
    SHAPE = 800
    CHANNELS = 1
   
if(GPU):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
else:
    device = "cpu"
net = Lenet(SHAPE,True,CHANNELS,affine=False)
net.to(device)
import torch.optim as optim
from tqdm import tqdm
criterion = nn.CrossEntropyLoss().to(device)
#criterion = nn.MSELoss()
optimizer = optim.Adam(net.parameters(), lr=0.001)
test_losses = []
training_losses = []
train_accuracies = []
test_accuracies = []

#PATH = "mnist_net_no_q_batch.pth"
#PATH = "mnist_net_no_q_no_batch.pth"
#PATH = "mnist_net_no_q_batch_affine_fixed.pth"
if(LOAD_PATH!=""):
    checkpoint = torch.load(LOAD_PATH)
    net.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    train_losses = checkpoint["train_losses"]
    test_losses = checkpoint["test_losses"]
    train_accuracies= checkpoint["train_accuracies"]
    test_accuracies = checkpoint["test_accuracies"]
    max_epoches = 20
else:
    max_epoches = 100
if(DATASET=="mnist"):
    max_epoches //= 2

if(TRAIN):
    net.train()
    for epoch in range(max_epoches):  # loop over the dataset multiple times
        running_loss = 0.0    
        running_test_loss = 0.0
        amount_train = 0
        amount_test = 0
        #if(epoch>5):
        #    USE_Q = True
        #ROUND_NUMBER = (Q_NUM/max_epoches)*(epoch)+0.001
        correct_count = 0
        total_count = 0    
        i = 0
        for data in tqdm(trainloader):
            i+=1
            if(ROUND):
                ROUND_NUMBER = (Q_NUM/(len(trainloader)*(max_epoches)))*(len(trainloader)*epoch+i)+0.001
            # get the inputs; data is a list of [inputs, labels]        
            inputs, labels = data
            if(GPU):
                inputs = inputs.type(torch.cuda.FloatTensor)        
                labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize
            outputs = net(inputs)
            #labels_new = F.one_hot(labels,num_classes=10).type(torch.cuda.FloatTensor)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
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
                if(GPU):
                    inputs = inputs.type(torch.cuda.FloatTensor)
                    labels = labels.type(torch.cuda.LongTensor)
                # zero the parameter gradients        
                # forward + backward + optimize
                outputs = net(inputs)
                #labels_new = F.one_hot(labels,num_classes=10).type(torch.cuda.FloatTensor)
                loss = criterion(outputs, labels) 
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
        
    torch.save(
        {
        "model_state_dict":net.state_dict(),
        "optimizer_state_dict":optimizer.state_dict(),
        "train_losses":training_losses,
        "test_losses":test_losses,
        "train_accuracies":train_accuracies,
        "test_accuracies":test_accuracies},SAVE_PATH)
else:
    running_test_loss = 0
    amount_test = 0
    total_count = 0
    correct_count = 0
    out_compare = []
    started = False
    num_samples = 0
    with torch.no_grad():
        for data in tqdm(testloader):
            # get the inputs; data is a list of [inputs, labels]        
            inputs, labels = data
            inputs = inputs.type(torch.cuda.FloatTensor)
            labels = labels.type(torch.cuda.LongTensor)
            # zero the parameter gradients        
            # forward + backward + optimize
            USE_Q = True
            outputs,quantize_outs = net(inputs)
            USE_Q = False
            outputs,normal_outs = net(inputs)
            if(started==False):
                started = True
                out_compare = [0 for _ in range(len(quantize_outs))]
            try:
                for i in range(len(quantize_outs)):            
                    quantize_outs[i] = quantize_outs[i]#.detach().cpu().numpy()
                    normal_outs[i] = normal_outs[i]#.detach().cpu().numpy()
                    #out_compare[i] += np.mean(np.power(quantize_outs[i]-normal_outs[i],2))
                    if(VIEW_DIM):
                        m = get_max(quantize_outs[i])
                        #print(m)
                        if(m<MIN_DIM):
                            MIN_DIM = m
                        if(m>MAX_DIM):
                            MAX_DIM = m
                        AVG_DIM.append(m)
                num_samples +=1
            except ValueError:
                break
            #labels_new = F.one_hot(labels,num_classes=len(classes)).type(torch.cuda.FloatTensor)
            loss = criterion(outputs, labels)   
            # print statistics
            running_test_loss += loss.item()
            amount_test += 1
            out_argmax = torch.argmax(outputs,axis=1)
            for idx,item in enumerate(out_argmax):
                if(labels[idx]==item):
                    correct_count+=1
                total_count+=1
    VIEW_DIM = False
    avg_w_dim = 0
    w_dim_amount = 0
    for name,data in net.named_parameters():
        if("gamma" in name or "beta" in name):
            continue
        dim = get_max(data)        
        avg_w_dim += dim
        print(name,dim)
        w_dim_amount += 1
    test_accuracy = correct_count/total_count
    print(f'{id_string} test_loss: {running_test_loss/amount_test:.3f} Test Acc: {test_accuracy}')
    out_compare = np.array(out_compare)
    with open("out-cifar.txt","w") as file:
        file.write(str(out_compare))
    print(MAX_DIM,MIN_DIM,np.array(AVG_DIM).mean())
    print("W-Dim: ",avg_w_dim/w_dim_amount)
    with open(f"{DATASET}_results.txt","a") as file:
        file.write(f"{SAVE_PATH} | W-Dim: {avg_w_dim/w_dim_amount} | A-Dim: {np.array(AVG_DIM).mean()} | MAX_DIM : {MAX_DIM} | Accuracy: {test_accuracy}\n")
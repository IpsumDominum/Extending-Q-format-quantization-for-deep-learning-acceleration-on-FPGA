from nltk.translate.bleu_score import sentence_bleu
from Dataset.translation_dataset import EnglishToGermanDataset
from Transformer.transfomer import TransformerTranslator
from Transformer.sub_layers import output_bit_amount,output_bit_num,get_max,TEST_WEIGHTS
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
import random

"""
Hyperparameters
"""
CUDA = True
PRINT_INTERVAL = 5000
VALIDATE_AMOUNT = 10
SAVE_INTERVAL = 5000

batch_size = 8
embed_dim = 64
num_blocks = 2
num_heads = 1  # Must be factor of token size
max_context_length = 1000

num_epochs = 1000
learning_rate = 1e-3

use_teacher_forcing = False

device = torch.device("cuda:0" if CUDA else "cpu")

"""
Dataset
"""
dataset = EnglishToGermanDataset(CUDA=CUDA)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True,generator=torch.Generator(device=device))
dataloader_test = torch.utils.data.DataLoader(
    dataset, batch_size=batch_size, shuffle=True,generator=torch.Generator(device=device)
)

"""
Model
"""
encoder_vocab_size = dataset.english_vocab_len
output_vocab_size = dataset.german_vocab_len
torch.set_default_tensor_type(torch.cuda.FloatTensor if CUDA else torch.FloatTensor)
model = TransformerTranslator(
    embed_dim, num_blocks, num_heads, encoder_vocab_size,output_vocab_size,CUDA=CUDA
).to(device)

"""
Loss Function + Optimizer
"""
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.KLDivLoss(reduction='batchmean')

"""
Load From Checkpoint
"""
LOAD = 235000

if LOAD != -1:
    checkpoint = torch.load(
        os.path.join("Checkpoints", "Checkpoint" + str(LOAD) + ".pkl")
    )
    test_losses = checkpoint["test_losses"]
    train_losses = checkpoint["train_losses"]
    num_steps = checkpoint["num_steps"]
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
else:
    test_losses = []
    train_losses = []
    num_steps = 0
"""
Validation LOOP
"""
dataset.test()
model.eval()
avg_errors = 0
avg_errors_wer = 0
avg_samples = 0

def get_wer(label,pred):
    correct = 0
    total = len(label)
    for i in range(max(len(label),len(pred))):
        if(i>=len(label)):
            label_tok = ""
        else:
            label_tok = label[i]
        if(i>=len(pred)):
            pred_tok = ""
        else:
            pred_tok = pred[i]
        if(pred_tok==label_tok):
            correct +=1
    return correct/total

with torch.no_grad():
    jdx = 0
    for item in tqdm(dataloader_test):
        model.encode(item["english"])
        all_outs = torch.tensor([], requires_grad=False).to(device)
        all_outs_tokens = item["german"][:,:1]
        for i in range(item["german"].shape[1] - 1):
            #No teacher forcing in validation                        
            out = model(all_outs_tokens[:,:i+1])
            out_token = torch.argmax(out,dim=1)
            all_outs = torch.cat((all_outs, out), dim=1)                        
            all_outs_tokens = torch.cat((all_outs_tokens,out_token),dim=1)
        all_outs = all_outs * item["logit_mask"][:,1:,:]
        item["logits"] = item["logits"] * item["logit_mask"]
        loss = criterion(all_outs, item["logits"][:,1:,:])
        for j in range(len(item["logits"])):
            label = dataset.logit_to_sentence_tokens(item["logits"][j])
            pred = dataset.logit_to_sentence_tokens(all_outs[j])            
            avg_errors += sentence_bleu([label],pred)          
            avg_errors_wer += get_wer(label,pred)          
            avg_samples += 1
        jdx +=1
        if(TEST_WEIGHTS):
            for name,data in model.named_parameters():
                print(name,get_max(data))
            exit()
print(avg_errors/avg_samples)
print(1-avg_errors_wer/avg_samples)

import matplotlib.pyplot as plt
import torch
import os

checkpoint = torch.load(os.path.join("Checkpoints", "Checkpoint235000.pkl"))
plt.plot(checkpoint["train_losses"],label="train")
plt.plot(checkpoint["test_losses"],label="test")
#checkpoint = torch.load(os.path.join("Checkpoints", "Checkpoint235000.pkl"))
#plt.plot(checkpoint["train_losses"])
#plt.plot(checkpoint["test_losses"])
plt.title("Transformer Train")
plt.legend(loc="upper left")
plt.show()

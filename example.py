import torch
import torch.nn as nn
import torch.optim as optim
from deltanet import DeltaBlock

"""
    this is an example on how to use the architecture. It's byte per byte text generation, it uses a dumb model without mlp bloc, it's capacity limited
"""

def text_to_tensor(texts):
    B = len(texts)  # Number of text samples
    L = max(len(text) for text in texts)  # Max length of the texts
    D = 256  # Number of possible ASCII characters (0-255 inclusive)
    out = torch.zeros((B, L, D))  # Create a zero tensor with shape (B, L, D)
    
    for b in range(B):  # Iterate over batch
        for c in range(len(texts[b])):  # Iterate over characters in the b-th text
            out[b, c, ord(texts[b][c])] = 1.0  # Set the corresponding ASCII index to 1.0
    
    return out
    
def tensor_to_text(tensor):
    B, L, D = tensor.shape  # Get the shape of the tensor
    texts = []
    
    for b in range(B):  # Iterate over batch
        text = ""
        for l in range(L):  # Iterate over sequence length
            char_idx = torch.argmax(tensor[b, l]).item()  # Get the index of the max value
            if char_idx != 0:  # Ignore padding (assuming 0 represents no character)
                text += chr(char_idx)  # Convert index back to character
        texts.append(text)  # Append reconstructed string to the list
    
    return texts

def tensor_to_char(tensor):
    index = torch.argmax(tensor)
    return chr(index)

def char_to_tensor(char):
    tensor = torch.zeros(256)
    tensor[ord(char)]=1.0
    return tensor

def generated(tensor):
    new_tensor = torch.zeros(256)
    index = torch.argmax(tensor)
    new_tensor[index] = 1.0
    return new_tensor

def generate(model,prompt,seq_len=10):
    S = None
    #print(prompt[0],end="")
    for c in prompt:
        x = char_to_tensor(c)
        x,S = model.step(x,S)
        print(c,end="")
        x = x.squeeze(0).squeeze(0)
    print(tensor_to_char(x),end="")
    for i in range(10):
        x,S = model.step(x,S)
        x = generated(x)
        x = x.squeeze(0).squeeze(0)
        print(tensor_to_char(x),end="")

inputs = text_to_tensor(["hello world","bonjour","test"])
target = inputs.roll(shifts=-1,dims=1)

model = DeltaBlock(256,2)

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(),lr=0.001)

num_epoch = 500

for epoch in range(num_epoch):
    optimizer.zero_grad()
    output = model(inputs)
    loss = criterion(output,target)
    loss.backward()
    optimizer.step()
    if epoch%10==0:
        print(f"epoch:{epoch}/ {num_epoch}, loss:{loss.item()}")

print(tensor_to_text(output))
generate(model,"h")


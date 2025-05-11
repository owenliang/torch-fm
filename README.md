# torch-fm

pytorch flow matching implementation

MNIST dataset is used for testing

# sample

```
import matplotlib.pyplot as plt

x=torch.randn(size=(1,1,28,28)).to(device)
steps=250
label=8

model.eval()
with torch.no_grad():
    for i in range(steps):
        t=torch.tensor([1.0/steps*i]).to(device)
        label=torch.tensor([label],dtype=torch.long).to(device)
        pred_vt=model(x,t,label)
        x=x+pred_vt*1.0/steps
        x=x.detach()
    
x=(x+1)/2
plt.figure(figsize=(1,1))
plt.axis('off')
plt.imshow(x[0,0].cpu().numpy(),cmap='gray')
```

![](sample.png)

# coding: utf-8

# # Model Experiments
# 

# In[30]:




# In[31]:


from PIL import Image as pil_image


# In[32]:


#pip3 install nvidia-ml-py3
import tracemalloc, threading, torch, time, pynvml
from fastai.utils.mem import *
from fastai.vision import *


# In[33]:


import fastai
print(fastai.__version__)


# In[34]:


torch.cuda.set_device(0)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# In[35]:


#see https://forums.fast.ai/t/gpu-optimizations-central/33944/15
#"memory profiler that taps into each epoch, and can be fine-tuned to each separate stage"
if not torch.cuda.is_available(): raise Exception("pytorch is required")

def preload_pytorch():
    torch.ones((1, 1)).cuda()
    
def gpu_mem_get_used_no_cache():
    torch.cuda.empty_cache()
    return gpu_mem_get().used

def gpu_mem_used_get_fast(gpu_handle):
    info = pynvml.nvmlDeviceGetMemoryInfo(gpu_handle)
    return int(info.used/2**20)

preload_pytorch()
pynvml.nvmlInit()

class PeakMemMetric(LearnerCallback):
    _order=-20 # Needs to run before the recorder

    def peak_monitor_start(self):
        self.peak_monitoring = True


# In[36]:


path = Path('../data/mnist/mnist_png')


# In[37]:


np.random.seed(42)


# ### Custom model

# In[38]:


#no transforms
tfms = get_transforms(do_flip=False, 
                      flip_vert=False, 
                      max_rotate=0., 
                      max_zoom=0., 
                      max_lighting=0., 
                      max_warp=0., 
                      p_affine=0., 
                      p_lighting=0.)


# In[39]:


data = ImageDataBunch.from_folder(path, valid_pct = 0.2, ds_tfms=tfms, size=28)


# In[40]:


data.show_batch(rows=3, figsize=(5,5))


# In[41]:


x,y = next(iter(data.train_dl))
x.shape,y.shape


# In[42]:


class Mnist_Logistic(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin = nn.Linear(784, 10, bias=True)

    def forward(self, xb): 
        xb = xb[:, 0, :, :]
        xb=xb.view(-1, 28*28)
        return self.lin(xb)


# TODO work how to load a greyscale only dataset in fastai. By default 3 channels are created. In above code we use only first channel

# In[43]:


model = Mnist_Logistic().to(device)


# In[44]:


model


# In[45]:


model.lin


# In[46]:


model(x).shape


# In[47]:


[p.shape for p in model.parameters()]


# In[48]:


loss_func=nn.CrossEntropyLoss()


# In[52]:


lr=2e-2


# In[53]:


def update(x,y,lr):
    wd = 1e-5
    #get output of model
    y_hat = model(x)
    # weight decay
    w2 = 0.
    for p in model.parameters(): 
        w2 += (p**2).sum()
    # add to regular loss
    loss = loss_func(y_hat, y) + w2*wd
    loss.backward()
    with torch.no_grad():
        for p in model.parameters():
            p.sub_(lr * p.grad)
            p.grad.zero_()
    return loss.item()


# In[54]:


losses = [update(x,y,lr) for x,y in data.train_dl]


# In[55]:


plt.plot(losses)


# ## 2 Layers

# In[56]:


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        xb = xb[:, 0, :, :]
        xb=xb.view(-1, 28*28)
        x = self.lin1(xb)
        x = F.relu(x)
        return self.lin2(x)


# In[57]:


model = Mnist_NN().to(device)


# In[58]:


losses = [update(x,y,lr) for x,y in data.train_dl]


# In[59]:


plt.plot(losses)


# In[60]:


model = Mnist_NN().to(device)


# In[61]:


def update(x,y,lr):
    opt = optim.Adam(model.parameters(), lr)
    y_hat = model(x)
    loss = loss_func(y_hat, y)
    loss.backward()
    opt.step()
    opt.zero_grad()
    return loss.item()


# In[62]:


losses = [update(x,y,1e-3) for x,y in data.train_dl]


# In[63]:


plt.plot(losses)


# ## 3 Layers

# In[64]:


class Mnist_NN(nn.Module):
    def __init__(self):
        super().__init__()
        self.lin1 = nn.Linear(784, 50, bias=True)
        self.lin2 = nn.Linear(50, 50, bias=True)
        self.lin3 = nn.Linear(50, 10, bias=True)

    def forward(self, xb):
        xb = xb[:, 0, :, :]
        xb=xb.view(-1, 28*28)
        x = self.lin1(xb)
        x = F.relu(x)
        x = self.lin2(x)
        x = F.relu(x)
        x = self.lin3(x)
        return x


# In[65]:


model = Mnist_NN().to(device)


# In[66]:


losses = [update(x,y,1e-3) for x,y in data.train_dl]


# In[67]:


plt.plot(losses)


# In[ ]:


## Resnet


# In[68]:


learn = create_cnn(data, models.resnet18, metrics=error_rate)


# In[73]:


learn.lr_find()
learn.recorder.plot()


# In[70]:


learn.fit(2)


# In[ ]:


learn = create_cnn(data, models.resnet18, metrics=error_rate)


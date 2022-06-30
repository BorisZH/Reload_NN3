import torch
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets
from torch.utils.data import DataLoader
import numpy
import matplotlib.pyplot as plt


#hyper params
num_epoch = 2
cuda_device = -1
batch_size = 128
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

#model

#conv encoder
class Encoder(nn.Module):
    # 28*28 -> hidden -> out
    def __init__(self, in_chan, hidden_ch):
        super().__init__()
        #conv2d -> maxpool2d -> conv2d -> maxpool2d -> conv2d
        pass

    def forward(self, x): # -> 7x7
        pass


# sampling
def sampling(mu, sigma):
    pass


class Decoder(nn.Module):
    #conv2d -> upsampling2d -> conv2d -> upsampling2d -> conv2d
    def __init__(self, in_chan, hidden_ch):
        super().__init__()
        pass

    def forward(self, x): # -> 28 x 28
        pass


class AutoEncoder(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super().__init__()
        pass

    def forward(self, x):
        pass


def collate_fn(data):
    pics = []
    target = []
    for item in data:

        pics.append(numpy.array(item[0]))
        target.append(item[1])
    return {
        'data': torch.from_numpy(numpy.array(pics)).float() / 255,
        'target': torch.from_numpy(numpy.array(target)),
    }


def kl_loss(mu, sigma):
    p = torch.distributions.Normal(mu, sigma)
    q = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(sigma))

    return torch.distributions.kl_divergence(p, q).mean()


# model
model = AutoEncoder(1, 50)
model.train()
model.to(device)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler

#dataset
dataset = datasets.MNIST('/Users/a14419009/Repos/NN_reload_stream2', download=False)


#loss
criterion = nn.MSELoss()
#dataloder

for epoch in range(2):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        data = batch['data'].to(device).unsqueeze(1)
        # data_noized = torch.clamp(data + torch.normal(torch.zeros_like(data), torch.ones_like(data)), 0., 1.)
        optim.zero_grad()
        predict, mu, sigma = model(data)
        #loss
        kl = kl_loss(mu, sigma)
        crit_loss = criterion(data, predict)
        loss = 0.1 * kl + crit_loss
        loss.backward()
        optim.step()
        if (step % 100 == 0):
            print('kl_loss: {}, criterion_loss: {}'.format(kl.item(), crit_loss.item()))
    print(f'epoch: {epoch}')

#
# test = dataset.data[784].unsqueeze(0).unsqueeze(0).float() / 255
# predict = model(test)
#
# # plt.imshow(test[0].view(28, 28).detach().numpy())
# # plt.show()
#
# plt.imshow(predict[0][0].detach().numpy())
# plt.show()
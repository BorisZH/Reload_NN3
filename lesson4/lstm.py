import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from lesson4.dataset import DatasetSeq, collate_fn

data_dir = '/raid/home/bgzhestkov/nn_reload2/lesson4/'
train_lang = 'en'

dataset = DatasetSeq(data_dir)

#hyper params
vocab_len = len(dataset.word_vocab) + 1
n_classes = len(dataset.target_vocab) + 1
n_chars = len(dataset.char_vocab) + 1
cuda_device = -1
batch_size = 100
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'

ds_item = dataset[156]
#model
decode_words = [k for k in dataset.word_vocab]
print([decode_words[i] for i in ds_item['data']])

#using GRUCell
class POS_predictor(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        pass

    def forward(self, x):  # B x T
        pass

#usng GRU
class POS_predictorV2(nn.Module):
    def __init__(self,
                 vocab_size: int,
                 emb_dim: int,
                 hidden_dim: int,
                 n_classes: int,
                 ):
        super().__init__()
        pass

    def forward(self, x): # B x T
        pass


model = POS_predictorV2(vocab_len, 200, 256, n_classes)
model.train()
model = model.to(device)

#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler


#loss
loss_func = nn.CrossEntropyLoss()
#dataloder
for epoch in range(20):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        optim.zero_grad()
        data = batch['data'].to(device)  # B x T
        pred = model(data)
        loss = loss_func(pred.view(-1, n_classes), batch['target'].view(-1).to(device))
        loss.backward()
        # if step % 5:
        optim.step()

        if step % 50:
            print(loss)
    print(epoch)
    torch.save({'model': model.state_dict()}, '/raid/home/bgzhestkov/nn_reload3/epoch_%d.pth.tar' % epoch)
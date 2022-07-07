import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import numpy

from lesson5.data import DatasetSeq2Seq, collate_fn_seq2seq

data_dir = '/raid/home/bgzhestkov/nn_reload3/lesson5/dataset_text.txt'

dataset = DatasetSeq2Seq(data_dir)

#hyper params
input_vocab_len = len(dataset.input_sequnces_vocab)
output_vocab_len = len(dataset.output_sequnces_vocab)

cuda_device = -1
batch_size = 32
device = f'cuda:{cuda_device}' if cuda_device != -1 else 'cpu'


class Encoder(nn.Module):
    def __init__(self, vocab_len: int, emb_size: int = 256, hidden_size: int = 256):
        pass

    def forward(self, x):
        pass


class Decoder(nn.Module):
    def __init__(self,
                 vocab_len: int,
                 eos_id: int,
                 emb_size: int = 256,
                 hidden_size: int = 256,
                 ):
        pass

    def forward(self, hidden_state, sequence):
        pass

class DateNormalizer(nn.Module):
    def __init__(self, input_vocab_len, output_vocab_len, emb_size, hidden_size, eos_id):
        pass

    def forward(self, x, sequence):
        pass

model = DateNormalizer(input_vocab_len, output_vocab_len, 256, 256, dataset.output_sequnces_vocab['#']).to(device)
#optimizer
optim = torch.optim.Adam(model.parameters(), lr=0.001)
#lr scheduler


#
# import matplotlib.pyplot as plt
# plt.imshow(dataset.data[0].detach().numpy())
# plt.show()
#loss
loss_func = nn.CrossEntropyLoss()
#dataloder
for epoch in range(20):
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=collate_fn_seq2seq,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
    )
    for step, batch in enumerate(dataloader):
        input_seq = batch['data'].to(device)
        target = batch['target'].to(device)
        optim.zero_grad()
        predict = model(input_seq, target[:, :-1])

        loss = loss_func(predict.reshape(-1, output_vocab_len), target[:, 1:].reshape(-1))
        loss.backward()
        optim.step()
        if (step % 100 == 0):
            test = torch.tensor([[1, 19, 16, 27, 21, 29, 21, 22, 30, 27, 31, 32, 4, 18, 18, 4, 5, 6, 7, 4, 8, 3, 3, 28, 4, 23, 11,
                      2]])
            bos_tensor = torch.tensor([[dataset.output_sequnces_vocab['~']]])
            result = [3, 4, 4, 8, 6, 7, 8, 6, 10, 10, 9, 2]
            with torch.no_grad():
                model.eval()
                test_predict = model(test, bos_tensor)
                model.train()
            print(loss, test_predict)

    print('Epoch {} finished'.format(epoch))
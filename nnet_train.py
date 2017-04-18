
import torch
import torch.autograd
import torch.nn as nn
import torch.nn.functional as func
import torch.optim as optim
from torch.autograd import Variable
import numpy as np

from nnet_data import load_data_npy


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.hidden = nn.Linear(25, 10)
        self.output = nn.Linear(10, 2)

    def forward(self, x):
        x = func.tanh(self.hidden(x))
        x = func.log_softmax(self.output(x))
        return x


def label_data(data, label):
    _data = data[:, :, 0]
    _labels = (np.zeros(_data.shape[1]*data.shape[2]) if label == 0
               else np.ones(_data.shape[1]*data.shape[2]))
    for k in range(1, data.shape[2]):
        _data = np.concatenate([_data, data[:, :, k]], axis=1)
    return _data, _labels

def load_data(dir, existing=None):
    if existing is None:
        data = load_data_npy(dir=dir)
        del data['SkewformulaEstr1n50']
        del data['SkewformulaEstr15n50']
        del data['NormformulaEstr1n50']
        del data['NormformulaEstr15n50']
        white_list = ['NormbootEstr1n50', 'NormbootEstr15n50']
        white_list = ['SkewbootEstr1n50', 'SkewbootEstr15n50']
        new_dict = {}
        for fname in data.keys():
            if fname in white_list:
                new_dict[fname] = data[fname]
        data = new_dict

        fnames = list(data.keys())
        print(fnames)
        labels_init = [1, 0]
        assert len(fnames) == len(labels_init)

        labels, data_all = np.zeros(0), np.zeros((25, 0))
        for i in range(len(fnames)):
            pair = label_data(data[fnames[i]], labels_init[i])
            data_all = np.concatenate([data_all, pair[0]], axis=1)
            labels = np.concatenate([labels, pair[1]])

        print('...Saving labeled data as {}'.format('./labeled_data.npz'))
        np.savez('labeled_data', data_all, labels)
    else:
        ftype = ''
        if '.npz' != existing[-4:]:
            ftype = '.npz'
        d = np.load(existing + ftype)
        d = list(d.items())
        data_all, labels = d[0][1], d[1][1]
    return data_all, labels


if __name__ == '__main__':
    # options
    cuda = True
    lr = 0.01
    momentum = 0.5
    epochs = 1000
    test_batch_size = 1000

    print('...Loading data')
    data, labels = load_data(dir='./data_npy/', existing=None)#'labeled_data')

    print('...Formatting and shuffling data')
    _data = np.vstack([data, labels]).T
    np.random.shuffle(_data)
    data, labels = _data.T[:-1, :], _data.T[-1, :]
    data_train, labels_train = data[:, :data.shape[1]//2], labels[:labels.shape[0]//2]
    data_test, labels_test = data[:, data.shape[1]//2:], labels[labels.shape[0]//2:]
    print('\t...training set: {}\ttest set: {}'.format(data_train.shape, data_test.shape))

    print('...Defining model')
    model = Net()
    if cuda:
        model.cuda()

    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    _data_train, _labels_train = torch.FloatTensor(data_train.T), torch.LongTensor(labels_train.T.astype('int'))
    _data_test, _labels_test = torch.FloatTensor(data_test.T), torch.LongTensor(labels_test.T.astype('int'))
    _data_train, _labels_train = Variable(_data_train), Variable(_labels_train)
    _data_test, _labels_test = Variable(_data_test, volatile=True), Variable(_labels_test, volatile=True)
    if cuda:
        _data_train, _labels_train = _data_train.cuda(), _labels_train.cuda()
        _data_test, _labels_test = _data_test.cuda(), _labels_test.cuda()

    print('...Training')
    for epoch in range(1, epochs + 1):
        print('\t...epoch {}'.format(epoch))
        # train
        model.train()
        optimizer.zero_grad()
        output = model(_data_train)
        loss = func.nll_loss(output, _labels_train)
        loss.backward()
        optimizer.step()

        #model.eval()
        test_loss = 0
        correct = 0
        output = model(_data_test)
        loss_test = func.nll_loss(output, _labels_test).data[0]
        pred = output.data.max(1)[1]  # get index of max log-prob
        correct += pred.eq(_labels_test.data).cpu().sum()

        test_loss = loss_test
        test_loss /= _labels_test.numel()
        print('\t...Test set: loss: {:.4f}, accuracy: {}/{} ({:.0f}%)\n'.format(
            test_loss, correct, _labels_test.numel(),
            100. * correct / _labels_test.numel()
        ))

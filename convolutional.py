import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import torch.optim as optim
from dataLoader import RawNetData

class Residual_block(nn.Module):

    def __init__(self, nb_filts_in, nb_filts_out):
        super(Residual_block, self, nb_filts_in, nb_filts_out).__init__()

        self.conv1 = nn.Conv1d(in_channels= nb_filts_in, out_channels= nb_filts_out, kernel_size=3, padding=1, stride=1)
        self.bn1 = nn.BatchNorm1d(num_features=nb_filts_in)
        self.lrelu = nn.LeakyReLU()

        self.conv2 = nn.Conv1d(in_channels=nb_filts_in, out_channels=nb_filts_out, padding=1, kernel_size=3, stride=1)
        self.bn2 = nn.BatchNorm1d(num_features=nb_filts_in)

        self.MaxPool = nn.MaxPool1d(3)

        def forward(self, x):

            out = self.conv1(x)
            out = self.bn1(out)
            out = self.lrelu (out)

            out = self.conv2(out)
            out = self.bn2(out)
            out = self.lrelu(out)

            out = self.MaxPool(out)

            return out

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # Strided-conv
        self.conv1 = nn.Conv1d(in_channels = 3, out_channels = 128, kernel_size = 3, padding = 0)
        self.bn1 = nn.BatchNorm1d(num_features=128)
        self.lrelu = nn.LeakyReLU()

        self.block0 = self._make_layer(nb_blocks=2, nb_filtsIn=128, nb_filts_out= 128)
        self.block1 = self._make_layer(nb_blocks=4, nb_filtsIn=128, nb_filts_out= 256)

        self.gru = nn.GRU(input_size = 1024, num_layers = 1024,batch_first=True)
        self.fc1_gru = nn.Linear(in_features = 128, out_features = 128)
        self.fc2_gru = nn.Linear(in_features = 1211, out_features = 1211, bias = True)

    def forward(self, x):

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.lrelu(x)

        x = self.block0(x)
        x = self.block1(x)

        x, _ = self.gru(x)
        code = self.fc1_gru(x)

        out = self.fc2_gru(code)

        return out


    def _make_layer(self, nb_blocks, nb_filtsIn, nb_filtsOut):
        layers = []

        for i in range(nb_blocks):
            layers.append(ResBlock(nb_filts_in=nb_filtsIn, nb_filts_out=nb_filtsOut))

        return nn.Sequential(*layers)



DIRECTORY = "/info/home/larcher/ATAL/2019/voxceleb1/dev/wav"
net = Net()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
dataSet = RawNetData(DIRECTORY)
data_loader = torch.utils.data.DataLoader(dataSet,
                                                  batch_size=1,
                                                  shuffle=True,

                                                  )
#dataLoader = RawNetDataLoder("/home/olive/voxceleb1/dev/wav")
criterion = nn.CrossEntropyLoss()

for epoch in range(2):  # loop over the dataset multiple times

    running_loss = 0.0

    for i, data in enumerate(data_loader):
        optimizer.zero_grad()
        inputs, labels = data

        outputs = net(inputs)
        # loss = criterion(outputs, labels)
        #
        #
        # running_loss += loss.item()
        # if i % 2000 == 1999:    # print every 2000 mini-batches
        #     print('[%d, %5d] loss: %.3f' %
        #           (epoch + 1, i + 1, running_loss / 2000))
        #     running_loss = 0.0

print('Finished Training')
# ssh transit.univ-lemans.fr
# skinner
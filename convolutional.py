import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import torch.optim as optim
from dataLoader import RawNetData

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


DIRECTORY = "/home/olive/voxceleb1/dev/wav"
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
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import torch.optim as optim
from modele_Data import RawNetData
import numpy as np

class RawNet(nn.Module):
    def __init__(self):
        super(RawNet, self).__init__()

        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels = 1,#1
			out_channels = 128,#128
			kernel_size = 3,#3
                        padding = 0,
                        stride = 3
        )

        self.bn = nn.BatchNorm1d(num_features = 128)
        self.gru = nn.GRU(input_size = 59049,
			hidden_size = 1)

        self.gru_fc1 = nn.Linear(1,1024)
        self.gru_fc2 = nn.Linear(1024,59049)
        self.bn_before_gru = nn.BatchNorm1d(num_features = 256)


        self.gru2 = nn.Linear( in_features = 1, out_features = 1211)


        self.conv2 = nn.Conv1d(in_channels = 128,
                               out_channels = 128,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1
        )


        self.bn2 = nn.BatchNorm1d(num_features=128)
        
        self.conv3_1_1 = nn.Conv1d(in_channels = 128,
                               out_channels = 256,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1
        )

        self.conv3_1 = nn.Conv1d(in_channels = 256,
                               out_channels = 256,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1
        )



        self.conv3_2 = nn.Conv1d(in_channels = 256,
                               out_channels = 256,
                               kernel_size = 3,
                               padding = 1,
                               stride = 1
        )

        self.conv3_3 = nn.Conv1d(in_channels = 128,
                               out_channels = 256,
                               kernel_size = 1,
                               padding = 0,
                               stride = 1
        )


        self.bn3_1_1 = nn.BatchNorm1d(num_features=256)
        self.bn3_1 = nn.BatchNorm1d(num_features=128)
        self.bn3_2 = nn.BatchNorm1d(num_features=256)


        self.mp = nn.MaxPool1d(3)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn(out)
        out = self.lrelu(out)

        print("shape conv 1 : ", out.shape)

        #-------- Block 1 --------------
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)
        out = self.mp(out)

        out = self.bn2(out)
        out = self.lrelu_keras(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)
        out = self.mp(out)

        print("shape resblock 1 : ", out.shape)

        #-------- Block 2 --------------
        
        out_identite = out
        out = self.conv3_1_1(out)
        out = self.bn3_2(out)
        out = self.lrelu_keras(out)
        out = self.conv3_2(out)
        out += self.conv3_3(out_identite)
        out = self.mp(out)

        out_identite = out
        out = self.bn3_1_1(out)
        out = self.lrelu_keras(out)
        out = self.conv3_1(out)
        out = self.bn3_2(out)
        out = self.lrelu_keras(out)
        out = self.conv3_2(out)
        out += out_identite
        out = self.mp(out)

        out_identite = out
        out = self.bn3_1_1(out)
        out = self.lrelu_keras(out)
        out = self.conv3_1(out)
        out = self.bn3_2(out)
        out = self.lrelu_keras(out)
        out = self.conv3_2(out)
        out += out_identite
        out = self.mp(out)

        out_identite = out
        out = self.bn3_1_1(out)
        out = self.lrelu_keras(out)
        out = self.conv3_1(out)
        out = self.bn3_2(out)
        out = self.lrelu_keras(out)
        out = self.conv3_2(out)
        out += out_identite
        out = self.mp(out)

        print("shape resblock 2 : ", out.shape)
        

        #-------- Gru --------------

        out = self.bn_before_gru(out)
        out = self.lrelu_keras(out)
        out = out.permute(0, 2, 1)
        #(batch, filt, time) >> (batch, time, filt)

        out, _ = self.gru(out)
        out = out[:,-1,:]
        code = self.fc1_gru(out)
        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)
        out = self.fc2_gru(code)

        out = self.gru_fc1(out)
        out = self.gru_fc2(out)

        print("shape output : ", out)

        return out

def train(model, train_loader, optimizer, device):
    criterion = nn.CrossEntropyLoss()
    for batch_idx, (data,target) in enumerate(train_loader):

        
        data = data.to(device)
        target = target.to(device)

        output = model(data)
        #print("output : ", output)
        #print("data shape : ", data.size())
        #print("target shape : ", target.size())
        #print("shape output : ",output.shape)

        optimizer.zero_grad()
        loss = criterion(output,target)
        print("loss :", loss)
        
        loss.backward()
        optimizer.step()



if __name__ == '__main__':

    DIRECTORY = "/info/home/larcher/ATAL/2019/voxceleb1/dev/wav"
    print(DIRECTORY)
    print("-----")
    dataset = RawNetData(DIRECTORY)
    print("test data dataset : ", type(dataset.__getitem__(0)[0]))
    print("test target dataset : ", type(dataset.__getitem__(0)[1]))

    data_loader = torch.utils.data.DataLoader(dataset,batch_size=120,shuffle=True,
                                               drop_last=True, num_workers=12)
    print(data_loader)
    print("-----")
    model = RawNet()
    print(model)
    
    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.0001)
    print("-----ff")
    train(model,data_loader,optimizer,device)

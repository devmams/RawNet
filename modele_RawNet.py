# -*- coding: utf-8 -*-
#
# Ce fichier est une partie du modèle RawNet.
# RawNet est un modèle de reconnaissance du locuteur se basant sur l'article suivant :
# page d'accueil : https://arxiv.org/pdf/1904.08104.pdf
#
# les données d'apprentissage ont été récupéré sur la page suivant:
# page d'accueil :http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1.html
#
# Il a y a eu des implémentations en TenserFlow à la page suivante :
# page d'accueil :https://github.com/Jungjee/RawNet
#
# Ce modèle est en cours de développement: Vous pouvez y contribuer en travaillant avec nous :
# Contact : ...

"""
Copyright 2020 Olive DOVI, Mamadou DIALLO, Maymouna SY, Yassine M'CHAAR
"""
import torch.nn as nn
import torch.nn.functional as F
import importlib
import torch
import torch.optim as optim
from modele_Data import RawNetData
import numpy as np
__license__ = "LGPL"
__author__ = "Olive DOVI, Mamadou Diallo, Maymouna SY, Yassine M'CHAAR"
__copyright__ = "Copyright 2020 Olive DOVI"
__maintainer__ = "Olive DOVI, Mamadou DIALLO, Maymouna SY, Yassine M'CHAAR"
__email__ = "mawuss_olive.dovi.etu@univ-lemans.fr"
__status__ = "developpement"
__docformat__ = 'reS'

class Residual_block(nn.Module):
    """
        création du Residual_block
    """
    def __init__(self, nb_filts, first = False):
        """
           initialisation du Residual_block
           :param nb_filts
                exemple 2

            chaque bloc pourra être répété plusieurs fois
        """
        super(Residual_block, self).__init__()
        self.first = first

        if not self.first:
            self.bn1 = nn.BatchNorm1d(num_features = nb_filts[0])
        self.lrelu = nn.LeakyReLU()
        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)

        self.conv1 = nn.Conv1d(in_channels = nb_filts[0],
	        out_channels = nb_filts[1],
		kernel_size = 3,
		padding = 1,
		stride = 1)
        self.bn2 = nn.BatchNorm1d(num_features = nb_filts[1])
        self.conv2 = nn.Conv1d(in_channels = nb_filts[1],
		out_channels = nb_filts[1],
		padding = 1,
		kernel_size = 3,
		stride = 1)

        if nb_filts[0] != nb_filts[1]:
            self.downsample = True
            self.conv_downsample = nn.Conv1d(in_channels = nb_filts[0],
			out_channels = nb_filts[1],
			padding = 0,
			kernel_size = 1,
			stride = 1)
        else:
            self.downsample = False
        self.mp = nn.MaxPool1d(3)

    """
    permet de créer un block de calcul composé d'un ensemble de
    (convolution, LSTM, GRU, BatchNorm, LeakyReLU ...)
    """
    def forward(self, x):
        identity = x
        if not self.first:
            out = self.bn1(x)
            out = self.lrelu_keras(out)
        else:
            out = x

        out = self.conv1(x)
        out = self.bn2(out)
        out = self.lrelu_keras(out)
        out = self.conv2(out)

        if self.downsample:
            identity = self.conv_downsample(identity)

        out += identity
        out = self.mp(out)
        return out

class RawNet(nn.Module):
    """
        création du model Rawnet
    """
    def __init__(self):
        """
           initialisation du model rawnet

           L'objet crée notre modèle d'apprentissage
        """
        super(RawNet, self).__init__()

        self.lrelu_keras = nn.LeakyReLU(negative_slope=0.3)
        self.conv1 = nn.Conv1d(in_channels = 1,#1
			out_channels = 128,#128
			kernel_size = 3,#3
                        padding = 0,
                        stride = 3
        )
        self.bn = nn.BatchNorm1d(num_features = 128)

        self.block1 = self._make_layer(nb_blocks = 2,
			nb_filts = [128,128],
			first = True)

        self.block2 = self._make_layer(nb_blocks = 4,
			nb_filts = [128,256])


        self.bn_before_gru = nn.BatchNorm1d(num_features = 256)
        self.gru = nn.GRU(input_size = 256,
			hidden_size = 1024,
                        num_layers = 1,
                        batch_first = True)

        self.gru_fc1 = nn.Linear(in_features = 1024,
                                 out_features = 1024)

        self.gru_fc2 = nn.Linear(in_features = 1024,
                                 out_features = 1211,
                                 bias = True)

    def forward(self, x):

        out = self.conv1(x)
        out = self.bn(out)
        out = self.lrelu_keras(out)
        #print("shape conv 1 : ", out.shape)

        #-------- Block 1 --------------

        out = self.block1(out)
        #print("shape resblock 1 : ", out.shape)

        #-------- Block 2 --------------

        out = self.block2(out)
        #print("shape resblock 1 & 2 : ", out.shape)


        #-------- Gru --------------

        out = self.bn_before_gru(out)
        out = self.lrelu_keras(out)
        out = out.permute(0, 2, 1)
        #(batch, filt, time) >> (batch, time, filt)

        out, _ = self.gru(out)

        out = out[:,-1,:]
        code = self.gru_fc1(out)

        code_norm = code.norm(p=2,dim=1, keepdim=True) / 10.
        code = torch.div(code, code_norm)

        #print("shape GRU : ",code.shape)

        out = self.gru_fc2(code)

        #print("shape output : ", out.shape)
        return out



    def _make_layer(self, nb_blocks, nb_filts, first = False):
        layers = []
        #def __init__(self, nb_filts, first = False):
        for i in range(nb_blocks):
            first = first if i == 0 else False
            layers.append(Residual_block(nb_filts = nb_filts,
			first = first))
            if i == 0: nb_filts[0] = nb_filts[1]

            return nn.Sequential(*layers)





def train(model, train_loader, optimizer, device):
    """
       permet de faire notre apprentissage
       :param train_loader le dataloader du train
       :param optimizer pour optimiser nos paramètres
       :param device permet d'utiliser le GPU pour le calcul

    """
    criterion = nn.CrossEntropyLoss()
    num_epochs = 5         # The number of times entire dataset is trained
    for epoch in range(num_epochs):

        for i, (data,target) in enumerate(train_loader):


            data = data.to(device)
            target = target.to(device)

            output = model(data)
            #print("target shape : ", output)
            #print("shape target : ", target.size())
            #print("target shape[0] : ", target)

            optimizer.zero_grad()
            loss = criterion(output,target)
            #print("loss :", loss)

            loss.backward()
            optimizer.step()
            if (i+1) % num_epochs == 0:                              # Logging
                with open("resultat.txt", "a") as myfile:
                    s = 'Epoch [' + str(epoch+1) + '/' + str(num_epochs) + ']' + 'Loss : ' + str(loss.item()) + '\n'
                    myfile.write(s)


if __name__ == '__main__':

    DIRECTORY = "/home/s185313/data/voxceleb1/dev/wav"
    dataset = RawNetData(DIRECTORY)
    data_loader = torch.utils.data.DataLoader(dataset,batch_size=30,shuffle=True,
                                               drop_last=True, num_workers=12)
    print("-----------------------------------------------------------")

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if cuda else 'cpu')
    model = RawNet().to(device)
    print(model)

    learning_rate = 0.001
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate,
                                 weight_decay=0.0001)
    print("-----------------------------------------------------------")
    train(model,data_loader,optimizer,device)

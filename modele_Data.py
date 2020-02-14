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
import torch
from torch.utils.data.dataset import Dataset
import numpy as np
from wave import Wave_write
import torchaudio
import os
__license__ = "LGPL"
__author__ = "Olive DOVI, Sy Maymouna, Mamadou Diallo,yassine Macchar "
__copyright__ = "Copyright 2020 Olive DOVI"
__maintainer__ = "Olive DOVI, Mamadou DIALLO, Maymouna SY, Yassine M'CHAAR"
__email__ = "mawuss_olive.dovi.etu@univ-lemans.fr"
__status__ = "developpement"
__docformat__ = 'reS'




## permet de transformer la durée de l'audio de soorte à avoir 3,59 seconde
def transform_audio(audioTensor):
    """
    permet d'uniformiser la durée d'un audio pour avoir un audio de 3,59 secondes en sortie
    si la durée est:
        inférieur, on multiplie les fréquence pour avoir la taille neccessaire .
        supérieur, on le réduit ç la taille neccessaire
        :param audio de type Tensor
        :return: audio de type Tensor à la taille 3,59
    """
    nb_time = 59049
    x = audioTensor
    x = np.asarray(x[:, 1:] - 0.97 * x[:, :-1], dtype=np.float32)
    my_time = x.shape[1]
    if my_time > nb_time:
        start_idx = np.random.randint(low=0,
                                      high= my_time - nb_time)
        x = x[:, start_idx:start_idx + nb_time]
    elif my_time < nb_time:
        nb_dup = int(nb_time / my_time) + 1
        x = np.tile(x, (1, nb_dup))[:, :nb_time]
    else:
        x = x

    return x

def getFirst(directory):
    import os
    for _, d, _ in os.walk(directory):
        return d

def loadAudioFromDirectory(path):
    # get all wave files in the directory
    files = []
    for r, _, f in os.walk(path):
        for file in f:
            files.append(os.path.join(r, file))
        return files


# cette fonction permet de recuperer tous les fichiers audio d'un dossier wav et les classe par ID
# retourne une liste de liste, pour chaque element de la liste , le premier element correspond à l'id et le second correspond au chemin d'accès du fichier wave
def getAllWaveFileByWaveDirectory(directory):
    """
       permet de recupérer tous les audios wav present dans le repertoire classé par auteur
           :param directory repertoire wav
                exemple .../voxceleb1/dev/wav
           :return: listeWavefiles ,
                exemple : [[id1, ".../audio1.wav"], [id2, ".../audio2.wav"]]
    """
    listeWavefiles = []
    listeIdAuteur = []
    listeIdAuteur = getFirst(directory)[0:10];
    for idAuteur in listeIdAuteur:
        waveFolderByIdAuteur =getFirst(directory+"/"+idAuteur)
        for elt in waveFolderByIdAuteur:
            for file in  loadAudioFromDirectory(directory+"/"+idAuteur+"/"+elt):
                listeWavefiles.append([idAuteur, file])
    return listeWavefiles


class RawNetData(Dataset):
    """
        création du dataSet
    """

    def __init__(self,  directory):
        """
           initialisation du dataSet
           :param directory repertoire wav
                exemple .../voxceleb1/dev/wav

            l'objet dispose d'un attribut data avec la structure data, une liste contenant pour chaque elemment, l'auteur(son id ) et l'audio (chemin d'accès)
                exemple : [[id1, ".../audio1.wav"], [id2, ".../audio2.wav"]]
        """

        self.data = getAllWaveFileByWaveDirectory(directory)
        self.tab_id = []

    def __len__(self):
        # renvoie le nom d'element dans la liste wave
        return len(self.data )

    def __getitem__(self, idx):
        X , _ = torchaudio.load(self.data[idx][1])
        data= transform_audio(X)
        target = self.data[idx][0]
        if target in self.tab_id:
            return data, self.tab_id.index(target)
        else:
            self.tab_id.append(target)
            return data, self.tab_id.index(target)

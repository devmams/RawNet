class RawNetDataset(Dataset):
    def __init__(self, transform=None, mode="train",files_dir=None, base_dir="",csv_file_dir="",nb_time=59049):
        self.base_dir = base_dir
        self.mode = mode
        self.transform = transform
        self.csv_file_dir = csv_file_dir
        self.files_dir = files_dir
        csv_file = pd.read_csv(csv_file_dir,sep="\t")
        self.nb_time = nb_time
        self.classes = {cls_name:i+1 for i, cls_name in enumerate(csv_file["VoxCeleb1 ID"].unique())}

    def __len__(self):
        return len(self.files_dir)
    
    def __getitem__(self, idx):
        filename = self.files_dir[idx]
        classe = filename.split('/')[1]
        X, sample_rate = torchaudio.load(self.base_dir + filename)
        #print(" shape(X): ",type(X))
        label = self.classes[classe]
        self._pre_emphasis(X)
        nb_time = X.shape[1]
        if nb_time > self.nb_time:
            start_idx = np.random.randint(low = 0,
                high = nb_time - self.nb_time)
            X = X[:, start_idx:start_idx+self.nb_time]
            #print("nb_time: ",nb_time )
            #print("self.nb_time: ",self.nb_time)
        elif nb_time < self.nb_time:
            nb_dup = int(self.nb_time / nb_time) + 1
            X = np.tile(X, (1, nb_dup))[:, :self.nb_time]
            #print("taille inférieure")
        else:
            X = X
            #print("taille égale")
        #print(" type(X): ",X.size())

        #print('------------------------------------------------')

        return X, label

    def _pre_emphasis(self, x):
        '''
        Pre-emphasis for single channel input
        '''
        return np.asarray(x[:,1:] - 0.97 * x[:, :-1], dtype=np.float32) 


base_dir = '/info/home/larcher/ATAL/2019/voxceleb1/dev/wav'
csv_file_dir = 'vox1_meta.csv'
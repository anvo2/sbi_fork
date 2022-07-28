import torch
import torch.nn as nn 
import torch.nn.functional as F 
from torch.utils.data import Dataset
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
import pickle

class DatasetTorch(torch.utils.data.Dataset):
    '''
    A class that reads pickled dictionaries containing raw ddm data output of the form:
    {
      id1: {'data': (n_obs_max, 2), 'labels': (4,)}
      id2: {'data', 'labels'}
      }
    
    where n_obs_max is the maximum number of observation allowed, and labels indicates the number of parameters of the relevant simulator (usually 4 or 5 for ddm)
          data has form (RT, choice)
    each idx therefore contains enough simulated data to produce one batch of data for SBI, at n_obs_max.
    the dataloader will take in a batch_size, determine how many batches the read-in pickle file contains, then based on n_obs, will randomly draw
    (without replacement) n_obs from each batch, which then goes into SBI.
    
    When the current pickle file does not have enough data left for another batch, it reads in a new pickle file.
    
    
    ----
    path (str)          : path to the directory
    file_IDs (list)     : a list of all the names in the directory that contains the pickle files
    total_batches (int) : len of the dataloaders
    '''
    def __init__(self, 
                file_IDs,
                path,
                total_batches,
                batch_size = 32,
                n_obs = np.random.randint,
                label_prelog_cutoff_low = 1e-7,
                label_prelog_cutoff_high = None,
                device = 'cuda'
                ):

        # Initialization
        self.path = path
        self.batch_size = batch_size
        self.file_IDs = file_IDs
        self.n_obs = n_obs
        self.total_batches = total_batches
        self.indexes = np.arange(len(self.file_IDs))
        self.label_prelog_cutoff_low = label_prelog_cutoff_low
        self.label_prelog_cutoff_high = label_prelog_cutoff_high
        self.tmp_data = None
        self.device = device

        # get metadata from loading a test file

        self.__init_file_shape()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return self.total_batches
    
    def __init_file_shape(self):
        full_path = self.path + self.file_IDs[0]
        init_file = pickle.load(open(full_path, 'rb'))
        #print('Init file shape: ', init_file['data'].shape, init_file['labels'].shape)
        
        '''
        file_shape_dict: collects the length of the dictionary, which is then used to determine how many batches a file contains.
        '''
        self.file_shape_dict = len(init_file)
        self.batches_per_file = int(self.file_shape_dict / self.batch_size)
        self.input_dim = init_file[0]['data'].shape
        self.label_dim = init_file[0]['labels'].shape
        return
    
    def __load_file(self, file_index):
        full_path = self.path + self.file_IDs[file_index]
#         print('About to load file')
#         print(full_path)
        self.tmp_data = pickle.load(open(full_path, 'rb'))
        #shuffle_idx = np.random.choice(self.tmp_data['data'].shape[0], size = self.tmp_data['data'].shape[0], replace = True)
        #self.tmp_data['data'] = self.tmp_data['data'][shuffle_idx, :]
        #self.tmp_data['labels'] = self.tmp_data['labels'][shuffle_idx]
        return
    
    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        '''
        index: index of the batch
        
        '''
        # Find list of IDs
        # if 
        '''
        If index does not fully divide batchesper_file, then we don't have enough batches left in the dictionary to form a full batch, so load in a new one.
        '''
        if index % self.batches_per_file == 0 or self.tmp_data == None:
            self.__load_file(file_index = self.indexes[index // self.batches_per_file])

        # Generate data
        '''
        batch_ids give us a list of keys of the dictionary to pull (note that every idx in the dictionary is a batch. For example,
        with index = 5, batches_per_file = 10, batch_size = 5, this gives us [25, 26, 27, 28, 29], so we will pull the values of k = [25,26,27,28,29] from the
        dictionary
        '''
        batch_ids = np.arange(((index % self.batches_per_file) * self.batch_size), ((index % self.batches_per_file) + 1) * self.batch_size, 1)
        X, y = self.__data_generation(batch_ids)
        return X, y
    
    def __data_generation(self, batch_ids = None):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
#         objects = []
        
#         #Supply list of file name as a string
#         empty = ''
#         file_name = empty.join(self.file_IDs)
#         with (open(file_name, "rb")) as f:
#             while True:
#                 try:
#                     objects.append(pickle.load(f))
#                 except EOFError:
#                     break
#         #This collects the k,v of the dictionary based on batch_id
        newdict = {key: self.tmp_data[key] for key in batch_ids}
        if type(self.n_obs) is int:
            n_obs = self.n_obs
        else:
            n_obs = self.n_obs(100,500)
        data = []
        labels = []
        #Loop through every k,v pair and pull out n_obs
        
        for k in newdict:
            indices = np.random.choice(a = len(newdict[k]['data']),
                                   size = n_obs,
                                   replace = False)
            data.append (newdict[k]['data'][indices])
            labels.append(newdict[k]['labels'])
        
        if self.device == 'cuda':
            dataT = torch.tensor(np.array(data),dtype=torch.float, device = 'cuda')
            labelT = torch.tensor(np.array(labels), dtype = torch.float, device = 'cuda')
        else:
            dataT = torch.tensor(np.array(data),dtype=torch.float)
            labelT = torch.tensor(np.array(labels), dtype = torch.float)
        #X = torch.tensor(self.tmp_data['data'][batch_ids, :]) #tmp_file[batch_ids, :-1]
        #y = torch.unsqueeze(torch.tensor(self.tmp_data['labels'][batch_ids]),1) #tmp_file[batch_ids, -1]
        
        #if self.label_prelog_cutoff_low is not None:
            #y[y < np.log(self.label_prelog_cutoff_low)] = np.log(self.label_prelog_cutoff_low)
        
        #if self.label_prelog_cutoff_high is not None:
            #y[y > np.log(self.label_prelog_cutoff_high)] = np.log(self.label_prelog_cutoff_high)

        #return labels, data
        return labelT, dataT

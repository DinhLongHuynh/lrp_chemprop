import torch
from torch.utils.data import IterableDataset
import pandas as pd
from sklearn.preprocessing import StandardScaler
from Data_Preprocessor import Data_Preprocessor

class IterableMolDatapoints(IterableDataset):
    '''A class to prepare data for streaming, which is a subclass of IterableDataset. 
    The output is a generator that yields one chemprop.data.datasets.Datum at a time.
    '''

    def __init__(self, df, smiles_column, target_column, weight_column, scaler = None, size_at_time=100, shuffle=True):
        '''Parameters:
        ----------
        df (pd.DataFrame): A pandas dataframe containing the data.
        smiles_column (str): The column name containing SMILES strings.
        target_column (str): The column name containing the target values.
        scaler (StandardScaler): A StandardScaler object (already fitted) for normalizing the target values.
        size_at_time (int): The number of samples to transfrom into chemprop.data.datasets.Datum at a time.
        shuffle (boolean): If the df is shuffled.'''
        
        super().__init__()
        self.df = df
        self.smiles_column = smiles_column
        self.target_column = target_column
        self.weight_column = weight_column
        self.size_at_time = size_at_time
        self.shuffle= shuffle
        self.scaler = scaler

    def __len__(self):
        return len(self.df)

    def __iter__(self):
        '''A function to define iteration logic. It take the whole csv data, then shuffled, then access to only a subset of data at a time for transformation.
        The output is a generator that yields chemprop.data.datasets.Datum and ready to put through DataLoader.
        '''

        if self.shuffle:
            df_shuffled = self.df.sample(frac=1).reset_index(drop=True)
        else:
            df_shuffled = self.df.copy()

        # Transform pandas dataframe to molecule dataset according to size_at_time, prevent overloading memory. This is to balance between memory and speed.
        for i in range(0, len(df_shuffled), self.size_at_time):
            df_at_time = df_shuffled.iloc[i:i + self.size_at_time]
            data_generator = Data_Preprocessor()
            df_process = data_generator.generate(df=df_at_time,smiles_column=self.smiles_column,target_column=self.target_column,HB=True,weight_column='weight_lowscores')

            if self.scaler != None: 
                df_process.normalize_targets(scaler = self.scaler)

            # Handling parallelization manually
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is None: 
                for mol in df_process:
                    yield mol
            else: 
                num_workers = worker_info.num_workers
                worker_id = worker_info.id
                for i, mol in enumerate(df_process):
                    if i % num_workers == worker_id:
                        yield mol


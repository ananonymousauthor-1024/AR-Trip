import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch.utils.data.dataset import T_co


class ARDataset(Dataset):
    def __init__(self, df, pad_value=0):
        self.data = df
        self.pad_value = pad_value

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index) -> T_co:

        venue_ids = torch.tensor(self.data.iloc[index]['venue_ID'])
        hours_ids = torch.tensor(self.data.iloc[index]['hour'])
        latitudes = torch.tensor(self.data.iloc[index]['latitude'])
        longitudes = torch.tensor(self.data.iloc[index]['longitude'])
        # change hour 0 to 24
        hours_ids[hours_ids == 0] = 24

        # Mask the intermediate venue IDs and hour IDs (excluding start and end venues)
        mask_indices = torch.arange(1, len(venue_ids) - 1)
        masked_venue_ids = venue_ids.clone()
        masked_venue_ids[mask_indices] = self.pad_value
        masked_hour_ids = hours_ids.clone()
        masked_hour_ids[mask_indices] = self.pad_value
        masked_latitudes = latitudes.clone()
        masked_latitudes[mask_indices] = self.pad_value
        masked_longitudes = longitudes.clone()
        masked_longitudes[mask_indices] = self.pad_value

        return masked_venue_ids, masked_hour_ids, venue_ids, hours_ids, masked_latitudes, masked_longitudes

from brukerapi.dataset import Dataset
import numpy as np

def read_bruker(filename):

    # create Bruker data set
    d = Dataset(filename)

    # prepare data based on type of experiment
    if 'SPECTROSCOPY' in d.scheme._meta['id']:
        data = _prep_data_svs(d)
    elif 'CSI' in d.scheme._meta['id']:
        data = _prep_data_mrsi(d)

    # TODO general affine property in API
    return data, d['PVM_VoxelGeoCub'].affine, d.dwell_s

def _prep_data_svs(d):
    # add empty dimensions to push the temporal dimension to the 3rd index
    data = np.expand_dims(d.data, axis=(0,1,2))
    return data

def _prep_data_mrsi(d):
    # push the temporal dimension to the end
    data = np.transpose(d.data,(1,2,0))
    # add empty dimensions to push the temporal dimension to the 3rd index
    data = np.expand_dims(data, axis=2)
    return data
import h5py

def traverse_datasets(hdf_file):

    def h5py_dataset_iterator(g, prefix=''):
        for key in g.keys():
            item = g[key]
            path = '{0}/{1}'.format(prefix, key)  # f'{prefix}/{key}'
            if isinstance(item, h5py.Dataset):  # test for dataset
                yield (path, item)
            elif isinstance(item, h5py.Group): # test for group (go down)
                yield from h5py_dataset_iterator(item, path)

    with h5py.File(hdf_file, 'r') as f:
        for path, _ in h5py_dataset_iterator(f):
            yield path


# './trained_weights/VGG_ILSVRC_16_layers_fc_reduced.h5'
filename = './trained_weights/VGG_ssd300_Cityscapes/epoch-07_loss-36.9575_val_loss-51.2452.h5'
with h5py.File(filename, 'r') as hf:
    weights_names = []
    weights_shape = []
    for dset in traverse_datasets(filename):
        weights_names.append(dset)
        weights_shape.append(hf[dset].shape)
    fc6_weights = hf[weights_names[64]][:]
    conv5_3_weights = hf[weights_names[30]][:]
    pass

    # keys = list(hf.keys())
    # fc6 = hf['fc6']
    # fc6_keys = fc6.keys()
    # fc6_weights = fc6.values()
    # conv5_3 = hf['conv5_3']
    # conv5_3_keys = conv5_3.keys()
    # conv5_3_weights = conv5_3.values()
    # names = []
    # for name in hf:
    #     names.append(name)
    # # weights = []
    # # for key in keys:
    # #     weights.append(hf[key])
    # print(keys)


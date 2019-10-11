import torch
import torch.utils.data as data


def create_dataloader(dataset, conf):
    '''create dataloader '''
    if conf.is_train:
        return data.DataLoader(
            dataset,
            batch_size=conf.batch_size,
            shuffle= conf.use_shuffle,
            num_workers=conf.n_workers,
            drop_last=True,
            pin_memory=True)
    else:
        return data.DataLoader(
            dataset, batch_size=1, shuffle=False, num_workers=1, pin_memory=True)


def create_dataset(conf):
    mode = conf.dataset_name

    if mode == 'Multi_LDR_HDR':
        from .multi_ldr_hdr_dataset import Multi_LDR_HDR_Dataset as D
    elif mode == 'Multi_LDR_HDR_01':
        from .multi_ldr_hdr_dataset_01 import Multi_LDR_HDR_Dataset as D
    elif mode == 'FiveK':
        from .fivek_dataset import FiveKDataset as D
    else:
        raise NotImplementedError('Dataset [{:s}] is not recognized.'.format(mode))

    dataset = D(conf)
    print('Dataset [{:s}] is created.'.format(dataset.__class__.__name__))
    return dataset



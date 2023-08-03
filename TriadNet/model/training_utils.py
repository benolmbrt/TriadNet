from argparse import ArgumentParser
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
import os
import torch
from TriadNet.dataloader.mri_datamodule import LightningMRIModule

"""
Define utilitary functions for the training and inference of models
"""


def initialize_dm(df, args, val_batch_size=1, num_workers=1):
    """
    Initialize the MRI datamodule
    :param df: pandas Dataframe
    :param args: hyper-paremeters for the training
    :param val_batch_size: number of images in the validation batches
    :param num_workers: number of workers
    :return:
    """

    dm = LightningMRIModule(df=df,
                           image_key=args.image_key,
                           channel_name=args.channel_name,
                           segm_name=args.segm_name,
                           batch_size=args.batch_size,
                           num_workers=num_workers,
                           save_to_folder=args.output_folder,
                           val_batch_size=val_batch_size)

    return dm


class CustomArgumentParser(ArgumentParser):
    def convert_arg_line_to_args(self, arg_line):
        if ':' in arg_line:
            ret = []
            if "#" in arg_line:
                arg_line, _ = arg_line.split('#')
            arg_name, args_list = arg_line.split(':')
            ret.append(arg_name)
            for arg in args_list.split():
                ret.append(arg)
            return ret
        else:
            if "#" in arg_line:
                arg_line, _ = arg_line.split('#')
            return arg_line


def get_early_stop_callback(monitor, mode, args):
    early_stop_callback = EarlyStopping(
        monitor=monitor,
        min_delta=0.0,
        patience=args.early_stopping_patience,
        verbose=args.early_stopping_verbose,
        mode=mode)
    return early_stop_callback


def get_checkpoint_callback(dirpath,
                            filename,
                            monitor,
                            mode):
    checkpoint_callback = ModelCheckpoint(dirpath=dirpath,
                                          filename=filename,
                                          verbose=True,
                                          save_top_k=5,
                                          monitor=monitor,
                                          mode=mode)
    return checkpoint_callback


def get_best_ckpt_from_mode_and_monitor(folder, mode, monitor):
    """
    Find the best checkpoint in the checkpoints folder
    :param folder:
    :param mode:
    :param monitor:
    :return:
    """
    ckpts = [elt for elt in os.listdir(folder) if '.ckpt' in elt]
    if len(ckpts) == 0:
        raise ValueError('No checkpoint found in ', folder)
    checkpoints = sorted(ckpts, key=lambda x: get_ckpt_quantity(x, monitor=monitor))
    if mode == 'max':
        ckpt = os.path.join(folder, checkpoints[-1])
    print(f'Found checkpoint {ckpt} among {ckpts}')
    return ckpt


def get_ckpt_quantity(name, monitor='dice'):
    if monitor == 'epoch':
        suff = name.split('epoch=')[1].split('_')[0]
        return int(suff)
    else :
        pref = '_val_' + monitor
        length = 6
        pos = name.find(pref) + len(pref) + 1
        return float(name[pos:pos + length])


def string_to_torch_device(str_device):
    if str_device is None:
        device = torch.device('cpu')
        print('WARNING: Running on cpu')
    elif str_device in [0, 1, 2, 3]:
        print(f'Using cuda device {str(str_device)}')
        device = torch.device('cuda:' + str(str_device))  # not hashable !
    else:
        raise ValueError('Unrecognized device :', str_device)
    return device


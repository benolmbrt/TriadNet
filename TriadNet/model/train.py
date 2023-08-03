import pandas as pd
import os
import shutil
import sys
from pytorch_lightning import Trainer, loggers
from pytorch_lightning.callbacks import LearningRateMonitor

from TriadNet.model.pl_model import PLModule
from TriadNet.model.training_utils import CustomArgumentParser, get_early_stop_callback, get_checkpoint_callback, initialize_dm

"""
Training script on the simulated dataset
"""

def parse_args(config_file):
    """
    Read hyper-parameters from the config file
    :param config_file:
    :return:
    """
    parser = CustomArgumentParser(fromfile_prefix_chars='@')

    # folders
    parser.add_argument('--output-folder', type=str, required=True)
    parser.add_argument('--overwrite', action='store_true')

    # losses
    parser.add_argument("--n-classes", type=int, default=2)

    # data
    parser.add_argument('--data-csv', type=str, default=None, required=True)
    parser.add_argument("--channel-name", type=str, default='image')
    parser.add_argument("--segm-name", type=str, default='mask')
    parser.add_argument("--image-key", type=str, default='image_id')

    # training parameters
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--max-epochs", type=int, default=25)
    parser.add_argument('--early-stopping-patience', type=int, default=5)
    parser.add_argument('--early-stopping-verbose', type=bool, default=True)
    parser.add_argument("--batch-size", type=int, required=True)

    # training modality
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--gpus", type=int, nargs='+', required=False, default=None)

    # PL params
    parser.add_argument('--strategy', type=str, default='dp')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1)
    parser.add_argument('--num_sanity_val_steps', type=int, default=2)

    args = parser.parse_args(['@' + config_file])

    # create / overwrite folder
    if os.path.isdir(args.output_folder):
        if args.overwrite:
            print('Overwriting training folder')
            shutil.rmtree(args.output_folder)
        else:
            if len(os.listdir(args.output_folder)) > 0:
                assert False, 'out directory is not empty, use --overwrite if you want to erase it.'
    os.mkdir(args.output_folder)

    args.in_channels = 1

    return args


def main(args, config_filename):

    # save config and augmentation file in output_folder
    shutil.copy(config_filename, os.path.join(args.output_folder, 'config.yaml'))
    args.monitor, args.mode = 'val_dice', 'max'
    early_stop_callback = get_early_stop_callback(args.monitor, args.mode, args)
    ckpt_folder = os.path.join(args.output_folder, 'checkpoints')

    ckpt_save_name =  '{epoch}_{val_loss:.4f}_{val_dice:.4f}'
    checkpoint_callback = get_checkpoint_callback(ckpt_folder, ckpt_save_name, args.monitor, args.mode)
    lr_callback = LearningRateMonitor(logging_interval='step')

    logs_folder = os.path.join(args.output_folder, 'logs')
    tb_logger = loggers.TensorBoardLogger(logs_folder, default_hp_metric=False)

    # define dataloader
    df = pd.read_csv(args.data_csv)
    dm = initialize_dm(df, args, num_workers=args.num_workers)

    args.df = df

    # define PL module
    model = PLModule(args)

    # launch training
    if args.gpus is None:
        trainer = Trainer(logger=tb_logger,
                          max_epochs=args.max_epochs,
                          min_epochs=5,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          callbacks=[checkpoint_callback, early_stop_callback, lr_callback],
                          amp_backend="native")
    else:
        trainer = Trainer(logger=tb_logger,
                          max_epochs=args.max_epochs,
                          min_epochs=5,
                          gpus=args.gpus,
                          num_sanity_val_steps=args.num_sanity_val_steps,
                          callbacks=[checkpoint_callback, early_stop_callback, lr_callback],
                          strategy=args.strategy,
                          amp_backend="native")

    trainer.fit(model, dm)


if __name__ == '__main__':
    config_filename = sys.argv[1]
    args = parse_args(config_filename)
    main(args, config_filename)

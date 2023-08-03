import pytorch_lightning as pl
from torch.utils.data import DataLoader
from typing import List, Tuple
from pandas import DataFrame
import os
import torchio as tio

import numpy as np
from joblib import Parallel, delayed
from torchio.data.subject import Subject
from torchio.transforms import SpatialTransform
"""
Define Lighting Data module and utils for loading Nifti files
"""


class LightningMRIModule(pl.LightningDataModule):
    def __init__(self,
                 df: DataFrame,
                 image_key: str,
                 channel_name: str,
                 batch_size: int,
                 num_workers: int,
                 segm_name: str = None,
                 save_to_folder: str = None,
                 val_batch_size: int = 1
                 ):
        """
        :param df: Pandas Dataframe containing the paths to images and ground truths
        :param image_key: name of the column containing the image ids
        :param channel_name: name of the column containing the path to images
        :param batch_size: number of images per batch
        :param num_workers: number of workers
        :param segm_name: name of the column containing the path to ground truth masks
        :param save_to_folder: save the csv to the model folder
        :param val_batch_size: number of images per batch for validation steps
        """
        super().__init__()
        self.save_to_folder = save_to_folder
        self.df = df
        self.image_key = image_key
        self.channel_name = channel_name
        self.segm_name = segm_name
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_batch_size = val_batch_size

    def setup(self, stage=None):

        if stage is None or stage == 'fit':
            train_df, val_df = split_test_train_val(self.df)
            print(f'Using {len(train_df)} training samples')

            # save df to training folder
            if self.save_to_folder:
                train_df_save_path = os.path.join(self.save_to_folder, 'train_data.csv')
                train_df.to_csv(train_df_save_path)
                val_df_save_path = os.path.join(self.save_to_folder, 'val_data.csv')
                val_df.to_csv(val_df_save_path)

            # define torchio dataset
            training_subjects = generate_torchio_dataset(df=train_df, image_name=self.image_key,
                                                         segm_name=self.segm_name, channel_name=self.channel_name,
                                                         num_workers=self.num_workers)
            training_transform = generate_torchio_tranforms() # define preprocessing/data augmentation for training images
            # instantiate a torchio SubjectsDataset object
            self.training_dataset = tio.SubjectsDataset(training_subjects, transform=training_transform, load_getitem=False)

            if len(val_df) > 0:
                # same process for validation images
                print(f'Using {len(val_df)} validation samples')
                validation_subjects = generate_torchio_dataset(df=val_df, image_name=self.image_key,
                                                           segm_name=self.segm_name, channel_name=self.channel_name,
                                                               num_workers=self.num_workers)
                validation_transform = generate_torchio_tranforms()
                self.validation_dataset = tio.SubjectsDataset(validation_subjects, transform=validation_transform, load_getitem=False)
            else:
                self.validation_dataset = None

        elif stage == 'test':
            # define torchio test SubjectsDataset
            test_subjects = generate_torchio_dataset(df=self.df, image_name=self.image_key,
                                                     channel_name=self.channel_name, segm_name=self.segm_name, num_workers=self.num_workers)
            test_transform = generate_torchio_tranforms()
            self.test_dataset = tio.SubjectsDataset(test_subjects, transform=test_transform)

    def train_dataloader(self):
        return DataLoader(self.training_dataset, batch_size=self.batch_size, num_workers=self.num_workers,
                              shuffle=True, worker_init_fn=self.workers_init)

    def val_dataloader(self):
        return DataLoader(self.validation_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.val_batch_size, num_workers=self.num_workers,
                          shuffle=False)

    def workers_init(self, worker_id):
        """
        That method avoids that all the workers load data with the same seed, what would result in applying the same
        augmentation on data.

        Parameters
        ----------
        worker_id: int
            Id of the worker.
        """
        np.random.seed()


def split_test_train_val(df: DataFrame) -> Tuple[DataFrame, DataFrame]:
    """
    Split the dataframe in 2 parts: training and validation, based on the "train" column
    :param df:
    :return:
    """

    train_df = df[df['train'] == 1]  # train data
    val_df = df[df['train'] == 0]  # val data

    return train_df, val_df


def return_subject(df: DataFrame,
                   idx: int,
                   image_name: str,
                   channel_name: str,
                   segm_name: str = None,
                   ) -> tio.Subject:
    """
    Create a Torchio Subject by reading the images stored in the dataframe, at position idx
    :param df: pandas Daframe
    :param idx: row number
    :param image_name: key containing the id of the image
    :param channel_name: key containing the path to the image
    :param segm_name: key containing the path to the ground truth mask
    :return:
    """
    subject_dict = {}
    subject_dict['id'] = df.iloc[idx][image_name]  # image id
    subject_dict['image'] = tio.ScalarImage(os.path.join(df.iloc[idx][channel_name]))
    if segm_name is not None:
        subject_dict['segm'] = tio.LabelMap(os.path.join(df.iloc[idx][segm_name]))

    subject = tio.Subject(subject_dict)
    return subject


def generate_torchio_dataset(df: DataFrame,
                             image_name: str,
                             channel_name: str,
                             segm_name: str = None,
                             num_workers: int=1,
                             ) -> List[tio.Subject]:
    """
    Generate a Torchio dataset from a pandas Dataframe
    :param df:
    :param image_name:
    :param channel_name:
    :param segm_name:
    :param num_workers:
    :return:
    """

    subjects = Parallel(n_jobs=num_workers)(delayed(return_subject)(df, i, image_name, channel_name,
                                                                    segm_name) for i in range(len(df)))

    return subjects


def generate_torchio_tranforms():
    """
    Generate a simple preprocessing pipeline based on z-normalization.
    No Data Augmentation is used here.
    :return:
    """
    preprocessings_functions = [Cast(), tio.ZNormalization(masking_method=tio.ZNormalization.mean)]
    preprocessing = tio.Compose(preprocessings_functions)
    return preprocessing


def concatenate_channels(patient_dict: dict):
    """
    Convert a torchio Subject dict to torch tensors
    :param patient_dict: dict containing torchio images
    :return:
    """

    # get img channels
    out_dict = {'image': patient_dict['image'][tio.DATA].float(),
                'id': patient_dict['id']}

    if 'segm' in patient_dict:
        y = patient_dict['segm'][tio.DATA].long()
        out_dict['segm'] = y

    return out_dict


class Cast(SpatialTransform):
    """
    Cast image to float32. To use torch dataloader with torchio, all sequences (image AND mask) need to have
    the same type. But sometimes segmentation are in uint. This causes a crash
    see https://github.com/fepegar/torchio/issues/375
    """
    def __init__(
            self,
            **kwargs,
    ):
        super().__init__(**kwargs)

    def apply_transform(self, subject: Subject) -> Subject:
        for image in self.get_images(subject):
            image.set_data(image.data.float())

        return subject

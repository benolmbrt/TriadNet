import os
from argparse import ArgumentParser
from TriadNet.generate_data.generator import ImageGenerator
import nibabel as nib
import pandas as pd
import numpy as np
from tqdm import tqdm

"""
Generate a simple binary segmentation task in 3D.
3 datasets are generated : 
    - Train and Test ID contain in-distribution images
    - Test OOD contains out-of-distribution images obtained by adding motion noise to the image with TorchIO 
"""


def parseargs():
    """
    Parse arguments
    """
    parser = ArgumentParser(description="Generate artificial 3d data for trainings/testing")
    parser.add_argument('--folder', help='dir where to save the files', required=True, type=str)
    parser.add_argument('--image-size', type=int, nargs='+', default=[64, 64, 64],
                        help='tuple of 3 ints corresponding to the shape of the image')

    args = parser.parse_args()
    assert len(args.image_size) == 3, 'You must provide 3 values for image size but you give {}'.format(args.image_size)
    return args


if __name__ == '__main__':

    args = parseargs()
    generator = ImageGenerator(image_size=args.image_size, means=[0, 0.5], stds=[0.1, 0.1])

    cwd = os.getcwd()
    total_path = os.path.join(cwd, args.folder)

    if not os.path.isdir(total_path):
        os.mkdir(total_path)

    splits = {'train': 500,
              'calibration': 500,
              'test': 500}

    for split in splits.keys():
        out_data = []
        n_samples = splits[split]
        print(f'Generate data for split {split}')
        for k in tqdm(range(n_samples)):
            img_name = f'scan_{split}_{str(k)}.nii.gz'
            gt_name = f'scan_{split}_{str(k)}_gt.nii.gz'
            img, gt = generator.generate_scan()
            path2img = os.path.join(total_path, img_name)
            path2gt = os.path.join(total_path, gt_name)
            nib.save(img, path2img)
            nib.save(gt, path2gt)

            line = {'image_id': img_name.replace('.nii.gz', ''),
                    'image': path2img,
                    'mask': path2gt}
            out_data.append(line)

        df = pd.DataFrame(out_data)
        path2csv = os.path.join(total_path, split + '.csv')

        if split == 'train':
            train_val = np.random.choice([True, False], size=len(df), p=[0.90, 0.1]).astype(np.uint8)
            df = df.assign(train=pd.Series(train_val).values)

        print('Saving csv to {}'.format(path2csv))
        df.to_csv(path2csv)











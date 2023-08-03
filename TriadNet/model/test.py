import os
from argparse import ArgumentParser
import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np

from TriadNet.model.pl_model import PLModule
from TriadNet.model.training_utils import get_best_ckpt_from_mode_and_monitor, string_to_torch_device, initialize_dm
from TriadNet.dataloader.mri_datamodule import concatenate_channels

"""
Test script for the predictive intervals (PI)
2 steps are performed:
    1. PI are calibrated on val data to find the corrective values
    2. PI are tested on test data
"""


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--run-folder', type=str, required=True, help='path to the trained model folder')
    parser.add_argument('--calib-csv', type=str, required=True, help='path to the CSV containing the paths to ID images')
    parser.add_argument('--test-csv', type=str, required=True, help='path to the CSV containing the paths to OOD images')
    parser.add_argument('--device', type=int, default=None, help='GPU ID')
    parser.add_argument('--num-workers', type=int, default=1)
    parser.add_argument('--coverage', type=float, default=0.9, help='Target coverage (90% by default)')
    parser.add_argument('--dev-run', default=False, action='store_true', help='If True, launch training and inference on 10 images')

    args = parser.parse_args()

    args.device = string_to_torch_device(args.device)
    return args


def load_pl_model(folder, device):
    # load model for inference
    monitor, mode = 'dice', 'max'
    checkpoint_dir = os.path.join(folder, 'checkpoints')
    ckpt = get_best_ckpt_from_mode_and_monitor(checkpoint_dir, mode=mode, monitor=monitor)  # checkpoint with the best val dice
    model = PLModule.load_from_checkpoint(ckpt)
    model.eval()
    model.to(device)

    return model


def compute_interval(model, x, foreground_classes=[1]):
    """
    Generate volume-wise predictive intervas for image x, for each class in foreground_classes
    :param model: trained triad net
    :param x: input image
    :param foreground_classes:  list of classes to compute intervals
    :return:
    """
    pred = model.net.prediction(x)  # N H W D
    predictions = {}
    for n in foreground_classes:
        bounds_n = pred[f'uncertainty_bounds_{n}']
        lower = (bounds_n == 3).sum().item()
        mean = lower + (bounds_n == 2).sum().item()
        upper = mean + (bounds_n == 1).sum().item()

        predictions[f'lower_vol_{n}'] = lower
        predictions[f'mean_vol_{n}'] = mean
        predictions[f'upper_vol_{n}'] = upper

    return predictions


def generate_paper_figure(out_folder, data, alpha, target_class):
    """
    Generate a visualization of the intervals (Figure 2 in paper)
    :param out_folder:
    :param data:
    :param alphas: corrective values found during calibration
    :param target_classes: classes for which intervals are desired
    :return:
    """
    fig = plt.figure(figsize=(3, 3))
    n_sub = len(data)
    xs = np.linspace(1, n_sub, n_sub).astype(np.uint16)
    colors = 'green'
    class_label = 'Foreground'
    target_score = 'vol'

    means = np.asarray([item[f'mean_{target_score}_{target_class}'] for item in data])
    lower = np.asarray([item[f'lower_{target_score}_{target_class}'] for item in data])
    up = np.asarray([item[f'upper_{target_score}_{target_class}'] for item in data])
    ys = np.asarray([item[f'true_{target_score}_{target_class}'] for item in data])

    # order
    order = np.argsort(ys)
    ordered_ys = np.take_along_axis(ys, order, axis=0)
    ordered_means = np.take_along_axis(means, order, axis=0)
    ordered_lower = np.take_along_axis(lower, order, axis=0)
    ordered_upper = np.take_along_axis(up, order, axis=0)

    lower_cal = np.maximum(0, ordered_lower - alpha)
    upper_cal = ordered_upper + alpha

    plt.plot(xs, ordered_means, c=colors, marker=".", label=f'Predicted', linewidth=0.5, markersize=2)
    plt.plot(xs, ordered_ys, c='black', linestyle='dashed', label=f'True', linewidth=1)
    plt.title(class_label)

    plt.fill_between(xs, lower_cal, upper_cal, alpha=.3, color='dimgrey', label='PI', linewidth=1)
    plt.xlabel('Subject ID')
    plt.ylabel('Number of voxels')
    plt.legend(loc="upper left")

    save_path = os.path.join(out_folder, 'paper_figure.png')
    plt.savefig(save_path, dpi=2000, bbox_inches='tight')
    plt.close()


def compute_pi_metrics(data, _class, alpha):
    # compute intervals width and coverage
    metrics = []
    estimated_values = ['vol']
    for i, target_value in enumerate(estimated_values):
        for item in data:  # for each patient
            lower_bound = item[f'lower_{target_value}_{_class}']
            upper_bound = item[f'upper_{target_value}_{_class}']
            true_val = item[f'true_{target_value}_{_class}']
            mean_val = item[f'mean_{target_value}_{_class}']
            lower_bound_cal = max(0, lower_bound - alpha)
            upper_bound_cal = max(mean_val, upper_bound + alpha)

            is_in = np.logical_and((true_val <= upper_bound_cal), (true_val >= lower_bound_cal))
            width = upper_bound_cal - lower_bound_cal
            mae = np.abs(mean_val - true_val)

            row = {'score': target_value, 'class': _class, 'w': width, 'is_in': int(is_in), 'lb': lower_bound_cal,
                   'y_pred': mean_val, 'ub': upper_bound_cal, 'y_true': true_val, 'mae': mae, 'alpha': alpha}
            metrics.append(row)

    return metrics


if __name__ == '__main__':
    args = parse_args()
    run_folder = args.run_folder

    # Stage 1 : calibrate PI
    calib_df = pd.read_csv(args.calib_csv)
    model = load_pl_model(run_folder, args.device)

    if args.dev_run:
        # sample 10 images for debug
        calib_df = calib_df.sample(10)

    # instantiate torch dataloaders
    calib_dm = initialize_dm(calib_df, model.hparams, num_workers=args.num_workers)
    calib_dm.setup(stage='test')
    calib_data_loader = calib_dm.test_dataloader()

    calibration_df = []
    for i, batch in enumerate(tqdm(calib_data_loader, 0)):
        image_dict = concatenate_channels(batch)
        x = image_dict['image']
        y = image_dict['segm']
        batch_size = len(x)
        for b in range(batch_size):
            xb = x[b][None, ...].to(args.device)
            yb = y[b][None, ...].to(args.device)
            pred_dict = compute_interval(model, xb)
            pred_dict['true_vol_1'] = y.sum().item()

            calibration_df.append(pred_dict)

    calibration_df = pd.DataFrame(calibration_df)
    mean = np.asarray(calibration_df[f'mean_vol_1'].tolist())
    lower = np.asarray(calibration_df[f'lower_vol_1'].tolist())
    upper = np.asarray(calibration_df[f'upper_vol_1'].tolist())
    true = np.asarray(calibration_df[f'true_vol_1'].tolist())
    cal_scores = np.maximum(true - upper, lower - true)

    # perform calibration -> corrective value to apply on the bounds of the interval
    a = 1 - args.coverage
    n = len(calib_df)
    qhat = np.quantile(cal_scores, np.ceil((n + 1) * (1 - a)) / n, interpolation='higher')
    print(f'Found qhat={qhat} for class 1')

    # now perform testing
    test_df = pd.read_csv(args.test_csv)
    test_dm = initialize_dm(test_df, model.hparams, num_workers=args.num_workers)
    test_dm.setup(stage='test')
    test_data_loader = test_dm.test_dataloader()
    test_df = []
    for i, batch in enumerate(tqdm(test_data_loader, 0)):
        image_dict = concatenate_channels(batch)
        x = image_dict['image']
        y = image_dict['segm']
        batch_size = len(x)
        for b in range(batch_size):
            xb = x[b][None, ...].to(args.device)
            yb = y[b][None, ...].to(args.device)
            pred_dict = compute_interval(model, xb)
            pred_dict['true_vol_1'] = y.sum().item()

            test_df.append(pred_dict)

    out_metrics = compute_pi_metrics(test_df, _class=1, alpha=qhat)
    generate_paper_figure(run_folder, test_df, alpha=qhat, target_class=1)

    out_metrics = pd.DataFrame(out_metrics)

    all_Ws = out_metrics['w'].tolist()
    all_in = out_metrics['is_in'].tolist()
    all_maes = out_metrics['mae'].tolist()
    avg_w = sum(all_Ws) / len(all_Ws)
    avg_cov = sum(all_in) / len(all_in)
    avg_mae = sum(all_maes) / len(all_maes)
    print(f'Avg Coverage for Volume: {avg_cov}')
    print(f'Avg Width for Volume: {avg_w}')
    print(f'Avg MAE for Volume: {avg_mae}')







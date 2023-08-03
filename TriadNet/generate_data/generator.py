import numpy as np
import nibabel as nib
from scipy.ndimage.morphology import distance_transform_edt, generate_binary_structure, binary_erosion, binary_dilation
import torchio as tio
import torch

"""
A simple 3d Nifti generator for a binary segmentation mask.
The images contain spheres (class 1) and background (class 0)
Data are generated using the RandomLabelsToImage function of TorchIO 
To simulate annotation uncertainty, the ground truth mask is randomly dilated or eroded 
"""

class ImageGenerator:
    """
    Generate geometric forms inside a 3d image.
    """
    def __init__(self,
                 image_size,
                 means,
                 stds,
                 allow_overlap=True,
                 min_spheres=1,
                 max_spheres=1,
                 max_size_object=64,
                 min_size_object=16):
        """

        :param image_size: size of the 3d image
        :param means: mean intensity for each class
        :param stds: std intensity for each class
        :param allow_overlap: allow intersection between spheres
        :param min_spheres: minimum number of spheres in an image
        :param max_spheres: maximum number of spheres in an image
        :param max_size_object: maximum diameter of spheres in an image
        :param min_size_object: minimum diameter of spheres in an image
        """

        self.image_size = image_size
        self.allow_overlap = allow_overlap
        self.max_spheres = max_spheres
        self.min_spheres = min_spheres
        self.min_size_object = min_size_object
        self.max_size_object = max_size_object
        self.type = type
        assert len(means) == 2, f'Expect 2 values for the means but got {len(means)}'
        assert len(stds) == 2, f'Expect 2 values for the stds but got {len(stds)}'
        self.means = np.array(means)
        self.stds = np.array(stds)
        self.transform = tio.RandomLabelsToImage(label_key='tissues', mean=self.means, std=self.stds)
        self.deformation = tio.RandomElasticDeformation(p=1)

    def generate_sphere(self, image, gt, center_loc, rayon):
        """
        Add a random number of spheres inside the input image
        :param image:
        :return:
        """
        mask = np.ones_like(image)

        mask[center_loc] = 0
        dist_to_center = distance_transform_edt(mask)
        sphere_mask = np.where(dist_to_center <= rayon, 1, 0).astype(np.uint8)

        intersection = (sphere_mask * image > 0).astype(np.uint8)
        sphere_mask[intersection == 1] = 0

        sub = tio.Subject(img=tio.LabelMap(tensor=torch.from_numpy(sphere_mask).unsqueeze(0)))
        transformed = self.deformation(sub)['img']
        sphere_mask = transformed[tio.DATA][0].numpy()

        image += sphere_mask
        gt += sphere_mask

    def generate_blank_image(self):
        return np.zeros(self.image_size).astype(np.uint8)

    def generate_header(self):
        return np.eye(4)

    def randomize(self, nb_of_spheres):
        # generate a random number of spheres in a 3D volume
        img = np.zeros(self.image_size)
        gt = np.zeros(self.image_size)
        h, w, d = img.shape
        for k in range(nb_of_spheres):
            x = np.random.randint(0, h)
            y = np.random.randint(0, w)
            z = np.random.randint(0, d)
            size = np.random.randint(self.min_size_object, self.max_size_object)
            self.generate_sphere(img, gt, (x, y, z), size)
        return img, gt

    def generate_scan(self):
        """
        Generate a training/test image including non-overlapping (by default) squares and spheres
        :return:
        """
        nb_of_spheres = np.random.randint(self.min_spheres, self.max_spheres+1)

        it = 1
        img = np.zeros(self.image_size)
        while not np.array_equal(np.unique(img), np.array([0, 1])):
            img, gt = self.randomize(nb_of_spheres)
            it += 1

        subject = tio.Subject(tissues=tio.LabelMap(tensor=torch.from_numpy(img).unsqueeze(0)),
                              gt=tio.LabelMap(tensor=torch.from_numpy(gt).unsqueeze(0)))
        transformed = self.transform(subject)  # convert masks to random intensities
        img = transformed['image_from_labels'][tio.DATA][0].numpy()
        gt = transformed['gt'][tio.DATA][0].numpy()

        rdm = np.random.rand(1)
        rdm_it = np.random.randint(1, 2)
        struct = generate_binary_structure(3, 3)
        if rdm < 0.3:  # erosion of the ground truth mask
            gt = binary_erosion(gt, struct, rdm_it).astype(np.uint8)
        elif rdm >= 0.3 and rdm < 0.7:  # dilation of the ground truth mask
            gt = binary_dilation(gt, struct, rdm_it).astype(np.uint8)
        else:
            pass

        # add background
        header = self.generate_header()
        img = nib.Nifti1Image(img, header)
        gt = nib.Nifti1Image(gt, header)

        return img, gt
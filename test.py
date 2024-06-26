import os
import sys
import cv2
import glob
import argparse
import numpy as np
import yaml
from PIL import Image
import torch.nn.functional as F
import safetensors.torch

import torch
from torchvision import transforms
import torch.utils.checkpoint

from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import set_seed
from diffusers import AutoencoderKL, PNDMScheduler, UniPCMultistepScheduler, DPMSolverMultistepScheduler, DDPMScheduler#, StableDiffusionControlNetPipeline
from diffusers.utils import check_min_version
from diffusers.utils.import_utils import is_xformers_available
from transformers import CLIPTextModel, CLIPTokenizer, CLIPImageProcessor

from pipelines.pipeline_xpsr import StableDiffusionControlNetPipeline
from myutils.misc import load_dreambooth_lora
from myutils.wavelet_color_fix import wavelet_color_fix, adain_color_fix
#from annotator.retinaface import RetinaFaceDetection
import math
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from torchvision import transforms
import lpips
import pyiqa
from pytorch_fid.fid_score import calculate_fid_given_paths
from skimage import io
from scipy.ndimage import convolve
from scipy.special import gamma
from DISTS_pytorch import DISTS

def rgb2ycbcr_pt(img, y_only=False):
    """Convert RGB images to YCbCr images (PyTorch version).
    It implements the ITU-R BT.601 conversion for standard-definition television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    Args:
        img (Tensor): Images with shape (n, 3, h, w), the range [0, 1], float, RGB format.
         y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        (Tensor): converted images with the shape (n, 3/1, h, w), the range [0, 1], float.
    """
    if y_only:
        weight = torch.tensor([[65.481], [128.553], [24.966]]).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + 16.0
    else:
        weight = torch.tensor([[65.481, -37.797, 112.0], [128.553, -74.203, -93.786], [24.966, 112.0, -18.214]]).to(img)
        bias = torch.tensor([16, 128, 128]).view(1, 3, 1, 1).to(img)
        out_img = torch.matmul(img.permute(0, 2, 3, 1), weight).permute(0, 3, 1, 2) + bias

    out_img = out_img / 255.
    return out_img
def _ssim_pth(img, img2):
    """Calculate SSIM (structural similarity) (PyTorch version).
    It is called by func:`calculate_ssim_pt`.
    Args:
        img (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
        img2 (Tensor): Images with range [0, 1], shape (n, 3/1, h, w).
    Returns:
        float: SSIM result.
    """
    c1 = (0.01 * 255)**2
    c2 = (0.03 * 255)**2

    kernel = cv2.getGaussianKernel(11, 1.5)
    window = np.outer(kernel, kernel.transpose())
    window = torch.from_numpy(window).view(1, 1, 11, 11).expand(img.size(1), 1, 11, 11).to(img.dtype).to(img.device)

    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=img.shape[1])  # valid mode
    mu2 = F.conv2d(img2, window, stride=1, padding=0, groups=img2.shape[1])  # valid mode
    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2
    sigma1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=img.shape[1]) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu2_sq
    sigma12 = F.conv2d(img * img2, window, stride=1, padding=0, groups=img.shape[1]) - mu1_mu2

    cs_map = (2 * sigma12 + c2) / (sigma1_sq + sigma2_sq + c2)
    ssim_map = ((2 * mu1_mu2 + c1) / (mu1_sq + mu2_sq + c1)) * cs_map
    return ssim_map.mean([1, 2, 3])
def img2tensor(img):
    img = (img / 255.).astype('float32')
    if img.ndim ==2:
        img = np.expand_dims(np.expand_dims(img, axis = 0),axis=0)
    else:
        img = np.transpose(img, (2, 0, 1))  # C, H, W
        img = np.expand_dims(img, axis=0)
    img = np.ascontiguousarray(img, dtype=np.float32)
    tensor = torch.from_numpy(img)
    return tensor
def estimate_aggd_param(block):
    """Estimate AGGD (Asymmetric Generalized Gaussian Distribution) parameters.
    Args:
        block (ndarray): 2D Image block.
    Returns:
        tuple: alpha (float), beta_l (float) and beta_r (float) for the AGGD
            distribution (Estimating the parames in Equation 7 in the paper).
    """
    block = block.flatten()
    gam = np.arange(0.2, 10.001, 0.001)  # len = 9801
    gam_reciprocal = np.reciprocal(gam)
    r_gam = np.square(gamma(gam_reciprocal * 2)) / (gamma(gam_reciprocal) * gamma(gam_reciprocal * 3))

    left_std = np.sqrt(np.mean(block[block < 0]**2))
    right_std = np.sqrt(np.mean(block[block > 0]**2))
    gammahat = left_std / right_std
    rhat = (np.mean(np.abs(block)))**2 / np.mean(block**2)
    rhatnorm = (rhat * (gammahat**3 + 1) * (gammahat + 1)) / ((gammahat**2 + 1)**2)
    array_position = np.argmin((r_gam - rhatnorm)**2)

    alpha = gam[array_position]
    beta_l = left_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    beta_r = right_std * np.sqrt(gamma(1 / alpha) / gamma(3 / alpha))
    return (alpha, beta_l, beta_r)
def compute_feature(block):
    """Compute features.
    Args:
        block (ndarray): 2D Image block.
    Returns:
        list: Features with length of 18.
    """
    feat = []
    alpha, beta_l, beta_r = estimate_aggd_param(block)
    feat.extend([alpha, (beta_l + beta_r) / 2])

    # distortions disturb the fairly regular structure of natural images.
    # This deviation can be captured by analyzing the sample distribution of
    # the products of pairs of adjacent coefficients computed along
    # horizontal, vertical and diagonal orientations.
    shifts = [[0, 1], [1, 0], [1, 1], [1, -1]]
    for i in range(len(shifts)):
        shifted_block = np.roll(block, shifts[i], axis=(0, 1))
        alpha, beta_l, beta_r = estimate_aggd_param(block * shifted_block)
        # Eq. 8
        mean = (beta_r - beta_l) * (gamma(2 / alpha) / gamma(1 / alpha))
        feat.extend([alpha, mean, beta_l, beta_r])
    return feat
def niqe(img, mu_pris_param, cov_pris_param, gaussian_window, block_size_h=96, block_size_w=96):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    ``Paper: Making a "Completely Blind" Image Quality Analyzer``
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip
    Note that we do not include block overlap height and width, since they are
    always 0 in the official implementation.
    For good performance, it is advisable by the official implementation to
    divide the distorted image in to the same size patched as used for the
    construction of multivariate Gaussian model.
    Args:
        img (ndarray): Input image whose quality needs to be computed. The
            image must be a gray or Y (of YCbCr) image with shape (h, w).
            Range [0, 255] with float type.
        mu_pris_param (ndarray): Mean of a pre-defined multivariate Gaussian
            model calculated on the pristine dataset.
        cov_pris_param (ndarray): Covariance of a pre-defined multivariate
            Gaussian model calculated on the pristine dataset.
        gaussian_window (ndarray): A 7x7 Gaussian window used for smoothing the
            image.
        block_size_h (int): Height of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
        block_size_w (int): Width of the blocks in to which image is divided.
            Default: 96 (the official recommended value).
    """
    assert img.ndim == 2, ('Input image must be a gray or Y (of YCbCr) image with shape (h, w).')
    # crop image
    h, w = img.shape
    num_block_h = math.floor(h / block_size_h)
    num_block_w = math.floor(w / block_size_w)
    img = img[0:num_block_h * block_size_h, 0:num_block_w * block_size_w]

    distparam = []  # dist param is actually the multiscale features
    for scale in (1, 2):  # perform on two scales (1, 2)
        mu = convolve(img, gaussian_window, mode='nearest')
        sigma = np.sqrt(np.abs(convolve(np.square(img), gaussian_window, mode='nearest') - np.square(mu)))
        # normalize, as in Eq. 1 in the paper
        img_nomalized = (img - mu) / (sigma + 1)

        feat = []
        for idx_w in range(num_block_w):
            for idx_h in range(num_block_h):
                # process ecah block
                block = img_nomalized[idx_h * block_size_h // scale:(idx_h + 1) * block_size_h // scale,
                                      idx_w * block_size_w // scale:(idx_w + 1) * block_size_w // scale]
                feat.append(compute_feature(block))

        distparam.append(np.array(feat))

        if scale == 1:
            img = imresize(img / 255., scale=0.5, antialiasing=True)
            img = img * 255.

    distparam = np.concatenate(distparam, axis=1)

    # fit a MVG (multivariate Gaussian) model to distorted patch features
    mu_distparam = np.nanmean(distparam, axis=0)
    # use nancov. ref: https://ww2.mathworks.cn/help/stats/nancov.html
    distparam_no_nan = distparam[~np.isnan(distparam).any(axis=1)]
    cov_distparam = np.cov(distparam_no_nan, rowvar=False)

    # compute niqe quality, Eq. 10 in the paper
    invcov_param = np.linalg.pinv((cov_pris_param + cov_distparam) / 2)
    quality = np.matmul(
        np.matmul((mu_pris_param - mu_distparam), invcov_param), np.transpose((mu_pris_param - mu_distparam)))

    quality = np.sqrt(quality)
    quality = float(np.squeeze(quality))
    return quality
def calculate_niqe(img, crop_border, params_path, input_order='HWC', convert_to='y', **kwargs):
    """Calculate NIQE (Natural Image Quality Evaluator) metric.
    ``Paper: Making a "Completely Blind" Image Quality Analyzer``
    This implementation could produce almost the same results as the official
    MATLAB codes: http://live.ece.utexas.edu/research/quality/niqe_release.zip
    > MATLAB R2021a result for tests/data/baboon.png: 5.72957338 (5.7296)
    > Our re-implementation result for tests/data/baboon.png: 5.7295763 (5.7296)
    We use the official params estimated from the pristine dataset.
    We use the recommended block size (96, 96) without overlaps.
    Args:
        img (ndarray): Input image whose quality needs to be computed.
            The input image must be in range [0, 255] with float/int type.
            The input_order of image can be 'HW' or 'HWC' or 'CHW'. (BGR order)
            If the input order is 'HWC' or 'CHW', it will be converted to gray
            or Y (of YCbCr) image according to the ``convert_to`` argument.
        crop_border (int): Cropped pixels in each edge of an image. These
            pixels are not involved in the metric calculation.
        input_order (str): Whether the input order is 'HW', 'HWC' or 'CHW'.
            Default: 'HWC'.
        convert_to (str): Whether converted to 'y' (of MATLAB YCbCr) or 'gray'.
            Default: 'y'.
    Returns:
        float: NIQE result.
    """
    # ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
    # we use the official params estimated from the pristine dataset.
    niqe_pris_params = np.load(os.path.join(params_path, 'niqe_pris_params.npz'))
    mu_pris_param = niqe_pris_params['mu_pris_param']
    cov_pris_param = niqe_pris_params['cov_pris_param']
    gaussian_window = niqe_pris_params['gaussian_window']

    img = img.astype(np.float32)
    if input_order != 'HW':
        img = reorder_image(img, input_order=input_order)
        if convert_to == 'y':
            img = to_y_channel(img)
        elif convert_to == 'gray':
            img = cv2.cvtColor(img / 255., cv2.COLOR_BGR2GRAY) * 255.
        img = np.squeeze(img)

    if crop_border != 0:
        img = img[crop_border:-crop_border, crop_border:-crop_border]

    # round is necessary for being consistent with MATLAB's result
    img = img.round()

    niqe_result = niqe(img, mu_pris_param, cov_pris_param, gaussian_window)

    return niqe_result
def _convert_output_type_range(img, dst_type):
    """Convert the type and range of the image according to dst_type.
    It converts the image to desired type and range. If `dst_type` is np.uint8,
    images will be converted to np.uint8 type with range [0, 255]. If
    `dst_type` is np.float32, it converts the image to np.float32 type with
    range [0, 1].
    It is mainly used for post-processing images in colorspace conversion
    functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The image to be converted with np.float32 type and
            range [0, 255].
        dst_type (np.uint8 | np.float32): If dst_type is np.uint8, it
            converts the image to np.uint8 type with range [0, 255]. If
            dst_type is np.float32, it converts the image to np.float32 type
            with range [0, 1].
    Returns:
        (ndarray): The converted image with desired type and range.
    """
    if dst_type not in (np.uint8, np.float32):
        raise TypeError(f'The dst_type should be np.float32 or np.uint8, but got {dst_type}')
    if dst_type == np.uint8:
        img = img.round()
    else:
        img /= 255.
    return img.astype(dst_type)
def reorder_image(img, input_order='HWC'):
    """Reorder images to 'HWC' order.
    If the input_order is (h, w), return (h, w, 1);
    If the input_order is (c, h, w), return (h, w, c);
    If the input_order is (h, w, c), return as it is.
    Args:
        img (ndarray): Input image.
        input_order (str): Whether the input order is 'HWC' or 'CHW'.
            If the input image shape is (h, w), input_order will not have
            effects. Default: 'HWC'.
    Returns:
        ndarray: reordered image.
    """

    if input_order not in ['HWC', 'CHW']:
        raise ValueError(f"Wrong input_order {input_order}. Supported input_orders are 'HWC' and 'CHW'")
    if len(img.shape) == 2:
        img = img[..., None]
    if input_order == 'CHW':
        img = img.transpose(1, 2, 0)
    return img
def to_y_channel(img):
    """Change to Y channel of YCbCr.
    Args:
        img (ndarray): Images with range [0, 255].
    Returns:
        (ndarray): Images with range [0, 255] (float type) without round.
    """
    img = img.astype(np.float32) / 255.
    if img.ndim == 3 and img.shape[2] == 3:
        img = bgr2ycbcr(img, y_only=True)
        img = img[..., None]
    return img * 255.
def bgr2ycbcr(img, y_only=False):
    """Convert a BGR image to YCbCr image.
    The bgr version of rgb2ycbcr.
    It implements the ITU-R BT.601 conversion for standard-definition
    television. See more details in
    https://en.wikipedia.org/wiki/YCbCr#ITU-R_BT.601_conversion.
    It differs from a similar function in cv2.cvtColor: `BGR <-> YCrCb`.
    In OpenCV, it implements a JPEG conversion. See more details in
    https://en.wikipedia.org/wiki/YCbCr#JPEG_conversion.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
        y_only (bool): Whether to only return Y channel. Default: False.
    Returns:
        ndarray: The converted YCbCr image. The output image has the same type
            and range as input image.
    """
    img_type = img.dtype
    img = _convert_input_type_range(img)
    if y_only:
        out_img = np.dot(img, [24.966, 128.553, 65.481]) + 16.0
    else:
        out_img = np.matmul(
            img, [[24.966, 112.0, -18.214], [128.553, -74.203, -93.786], [65.481, -37.797, 112.0]]) + [16, 128, 128]
    out_img = _convert_output_type_range(out_img, img_type)
    return out_img
def imresize(img, scale, antialiasing=True):
    """imresize function same as MATLAB.
    It now only supports bicubic.
    The same scale applies for both height and width.
    Args:
        img (Tensor | Numpy array):
            Tensor: Input image with shape (c, h, w), [0, 1] range.
            Numpy: Input image with shape (h, w, c), [0, 1] range.
        scale (float): Scale factor. The same scale applies for both height
            and width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
            Default: True.
    Returns:
        Tensor: Output image with shape (c, h, w), [0, 1] range, w/o round.
    """
    squeeze_flag = False
    if type(img).__module__ == np.__name__:  # numpy type
        numpy_type = True
        if img.ndim == 2:
            img = img[:, :, None]
            squeeze_flag = True
        img = torch.from_numpy(img.transpose(2, 0, 1)).float()
    else:
        numpy_type = False
        if img.ndim == 2:
            img = img.unsqueeze(0)
            squeeze_flag = True

    in_c, in_h, in_w = img.size()
    out_h, out_w = math.ceil(in_h * scale), math.ceil(in_w * scale)
    kernel_width = 4
    kernel = 'cubic'

    # get weights and indices
    weights_h, indices_h, sym_len_hs, sym_len_he = calculate_weights_indices(in_h, out_h, scale, kernel, kernel_width,
                                                                             antialiasing)
    weights_w, indices_w, sym_len_ws, sym_len_we = calculate_weights_indices(in_w, out_w, scale, kernel, kernel_width,
                                                                             antialiasing)
    # process H dimension
    # symmetric copying
    img_aug = torch.FloatTensor(in_c, in_h + sym_len_hs + sym_len_he, in_w)
    img_aug.narrow(1, sym_len_hs, in_h).copy_(img)

    sym_patch = img[:, :sym_len_hs, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, 0, sym_len_hs).copy_(sym_patch_inv)

    sym_patch = img[:, -sym_len_he:, :]
    inv_idx = torch.arange(sym_patch.size(1) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(1, inv_idx)
    img_aug.narrow(1, sym_len_hs + in_h, sym_len_he).copy_(sym_patch_inv)

    out_1 = torch.FloatTensor(in_c, out_h, in_w)
    kernel_width = weights_h.size(1)
    for i in range(out_h):
        idx = int(indices_h[i][0])
        for j in range(in_c):
            out_1[j, i, :] = img_aug[j, idx:idx + kernel_width, :].transpose(0, 1).mv(weights_h[i])

    # process W dimension
    # symmetric copying
    out_1_aug = torch.FloatTensor(in_c, out_h, in_w + sym_len_ws + sym_len_we)
    out_1_aug.narrow(2, sym_len_ws, in_w).copy_(out_1)

    sym_patch = out_1[:, :, :sym_len_ws]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, 0, sym_len_ws).copy_(sym_patch_inv)

    sym_patch = out_1[:, :, -sym_len_we:]
    inv_idx = torch.arange(sym_patch.size(2) - 1, -1, -1).long()
    sym_patch_inv = sym_patch.index_select(2, inv_idx)
    out_1_aug.narrow(2, sym_len_ws + in_w, sym_len_we).copy_(sym_patch_inv)

    out_2 = torch.FloatTensor(in_c, out_h, out_w)
    kernel_width = weights_w.size(1)
    for i in range(out_w):
        idx = int(indices_w[i][0])
        for j in range(in_c):
            out_2[j, :, i] = out_1_aug[j, :, idx:idx + kernel_width].mv(weights_w[i])

    if squeeze_flag:
        out_2 = out_2.squeeze(0)
    if numpy_type:
        out_2 = out_2.numpy()
        if not squeeze_flag:
            out_2 = out_2.transpose(1, 2, 0)

    return out_2
def _convert_input_type_range(img):
    """Convert the type and range of the input image.
    It converts the input image to np.float32 type and range of [0, 1].
    It is mainly used for pre-processing the input image in colorspace
    conversion functions such as rgb2ycbcr and ycbcr2rgb.
    Args:
        img (ndarray): The input image. It accepts:
            1. np.uint8 type with range [0, 255];
            2. np.float32 type with range [0, 1].
    Returns:
        (ndarray): The converted image with type of np.float32 and range of
            [0, 1].
    """
    img_type = img.dtype
    img = img.astype(np.float32)
    if img_type == np.float32:
        pass
    elif img_type == np.uint8:
        img /= 255.
    else:
        raise TypeError(f'The img type should be np.float32 or np.uint8, but got {img_type}')
    return img
def calculate_weights_indices(in_length, out_length, scale, kernel, kernel_width, antialiasing):
    """Calculate weights and indices, used for imresize function.
    Args:
        in_length (int): Input length.
        out_length (int): Output length.
        scale (float): Scale factor.
        kernel_width (int): Kernel width.
        antialisaing (bool): Whether to apply anti-aliasing when downsampling.
    """

    if (scale < 1) and antialiasing:
        # Use a modified kernel (larger kernel width) to simultaneously
        # interpolate and antialias
        kernel_width = kernel_width / scale

    # Output-space coordinates
    x = torch.linspace(1, out_length, out_length)

    # Input-space coordinates. Calculate the inverse mapping such that 0.5
    # in output space maps to 0.5 in input space, and 0.5 + scale in output
    # space maps to 1.5 in input space.
    u = x / scale + 0.5 * (1 - 1 / scale)

    # What is the left-most pixel that can be involved in the computation?
    left = torch.floor(u - kernel_width / 2)

    # What is the maximum number of pixels that can be involved in the
    # computation?  Note: it's OK to use an extra pixel here; if the
    # corresponding weights are all zero, it will be eliminated at the end
    # of this function.
    p = math.ceil(kernel_width) + 2

    # The indices of the input pixels involved in computing the k-th output
    # pixel are in row k of the indices matrix.
    indices = left.view(out_length, 1).expand(out_length, p) + torch.linspace(0, p - 1, p).view(1, p).expand(
        out_length, p)

    # The weights used to compute the k-th output pixel are in row k of the
    # weights matrix.
    distance_to_center = u.view(out_length, 1).expand(out_length, p) - indices

    # apply cubic kernel
    if (scale < 1) and antialiasing:
        weights = scale * cubic(distance_to_center * scale)
    else:
        weights = cubic(distance_to_center)

    # Normalize the weights matrix so that each row sums to 1.
    weights_sum = torch.sum(weights, 1).view(out_length, 1)
    weights = weights / weights_sum.expand(out_length, p)

    # If a column in weights is all zero, get rid of it. only consider the
    # first and last column.
    weights_zero_tmp = torch.sum((weights == 0), 0)
    if not math.isclose(weights_zero_tmp[0], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 1, p - 2)
        weights = weights.narrow(1, 1, p - 2)
    if not math.isclose(weights_zero_tmp[-1], 0, rel_tol=1e-6):
        indices = indices.narrow(1, 0, p - 2)
        weights = weights.narrow(1, 0, p - 2)
    weights = weights.contiguous()
    indices = indices.contiguous()
    sym_len_s = -indices.min() + 1
    sym_len_e = indices.max() - in_length
    indices = indices + sym_len_s - 1
    return weights, indices, int(sym_len_s), int(sym_len_e)
def cubic(x):
    """cubic function used for calculate_weights_indices."""
    absx = torch.abs(x)
    absx2 = absx**2
    absx3 = absx**3
    return (1.5 * absx3 - 2.5 * absx2 + 1) * (
        (absx <= 1).type_as(absx)) + (-0.5 * absx3 + 2.5 * absx2 - 4 * absx + 2) * (((absx > 1) *
                                                                                     (absx <= 2)).type_as(absx))





sys.path.append('xpsr')

# Will error if the minimal version of diffusers is not installed. Remove at your own risks.
check_min_version("0.18.0.dev0")

logger = get_logger(__name__, log_level="INFO")

def load_xpsr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention):
    if args.use_xpsr_light:
        from models.xpsr_light.unet_2d_condition import UNet2DConditionModel
        from models.xpsr_light.controlnet import ControlNetModel
    else:
        from models.xpsr.unet_2d_condition import UNet2DConditionModel
        from models.xpsr.controlnet import ControlNetModel
    # Load scheduler, tokenizer and models.
    scheduler = DDPMScheduler.from_pretrained(args.pretrained_model_path, subfolder="scheduler")
    text_encoder = CLIPTextModel.from_pretrained(args.pretrained_model_path, subfolder="text_encoder")
    tokenizer = CLIPTokenizer.from_pretrained(args.pretrained_model_path, subfolder="tokenizer")
    vae = AutoencoderKL.from_pretrained(args.pretrained_model_path, subfolder="vae")
    feature_extractor = CLIPImageProcessor.from_pretrained(f"{args.pretrained_model_path}/feature_extractor")
    unet = UNet2DConditionModel.from_pretrained(args.xpsr_model_path, subfolder="unet")
    controlnet = ControlNetModel.from_pretrained(args.xpsr_model_path, subfolder="controlnet")

    personalized_model_root = "checkpoints/personalized_models"
    if args.use_personalized_model and args.personalized_model_path is not None:
        if os.path.isfile(f"{personalized_model_root}/{args.personalized_model_path}"):
            unet, vae, text_encoder = load_dreambooth_lora(unet, vae, text_encoder, f"{personalized_model_root}/{args.personalized_model_path}", 
                                                           blending_alpha=args.blending_alpha, multiplier=args.multiplier)
        else:
            unet = UNet2DConditionModel.from_pretrained_orig(personalized_model_root, subfolder=f"{args.personalized_model_path}") # unet_disney

    # Freeze vae and text_encoder
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)
    unet.requires_grad_(False)
    controlnet.requires_grad_(False)

    # For mixed precision training we cast the text_encoder and vae weights to half-precision
    # as these models are only used for inference, keeping weights in full precision is not required.
    weight_dtype = torch.float32
    if accelerator.mixed_precision == "fp16":
        weight_dtype = torch.float16
    elif accelerator.mixed_precision == "bf16":
        weight_dtype = torch.bfloat16

    # Move text_encode and vae to gpu and cast to weight_dtype
    text_encoder.to(accelerator.device, dtype=weight_dtype)
    vae.to(accelerator.device, dtype=weight_dtype)
    unet.to(accelerator.device, dtype=weight_dtype)
    controlnet.to(accelerator.device, dtype=weight_dtype)

    if enable_xformers_memory_efficient_attention:
        if is_xformers_available():
            unet.enable_xformers_memory_efficient_attention()
            controlnet.enable_xformers_memory_efficient_attention()
        else:
            raise ValueError("xformers is not available. Make sure it is installed correctly")

    # Get the validation pipeline
    validation_pipeline = StableDiffusionControlNetPipeline(
        vae=vae, text_encoder=text_encoder, tokenizer=tokenizer, feature_extractor=feature_extractor, 
        unet=unet, controlnet=controlnet, scheduler=scheduler, safety_checker=None, requires_safety_checker=False,
    )
    #validation_pipeline.enable_vae_tiling()
    validation_pipeline._init_tiled_vae(encoder_tile_size=args.encoder_tiled_size, decoder_tile_size=args.decoder_tiled_size)

    return validation_pipeline


def main(args, enable_xformers_memory_efficient_attention=True,):
    accelerator = Accelerator(
        mixed_precision=args.mixed_precision,
    )

    # If passed along, set the training seed now.
    if args.seed is not None:
        set_seed(args.seed)


    # We need to initialize the trackers we use, and also store our configuration.
    # The trackers initializes automatically on the main process.
    if accelerator.is_main_process:
        accelerator.init_trackers("xpsr")

    pipeline = load_xpsr_pipeline(args, accelerator, enable_xformers_memory_efficient_attention)

    resize_preproc = transforms.Compose([
        transforms.Resize(args.process_size, interpolation=transforms.InterpolationMode.BILINEAR),
    ])
    img_preproc = transforms.Compose([
            transforms.ToTensor(),
        ])
                
    if accelerator.is_main_process:
        generator = torch.Generator(device=accelerator.device)
        if args.seed is not None:
            generator.manual_seed(args.seed)

        image_names = []
        #folders = os.listdir(args.image_path)
        #folders = ["DIV2K_valid_old"]
        folders = ["RealSR_CenterCrop", "DrealSR_CenterCrop"]
        #folders = ["DIV2K_valid_old", "RealSR_CenterCrop", "DrealSR_CenterCrop"]
        #folders = ["DIV2K_valid"]
        #folders = [ "RealSR_CenterCrop"]
        for folder in folders:
            image_names.extend(sorted(glob.glob(f'{args.image_path}/{folder}/LR/*.png'))[:])

        for image_name in image_names[:]:
            validation_image = Image.open(image_name).convert("RGB")

            txt_path = image_name.replace("/LR/", "/highlevel_prompt/").replace(".png", ".txt")
            fp = open(txt_path, "r")
            try:
                validation_prompt = fp.readlines()[0].lstrip()
            except:
                validation_prompt = ""
            validation_prompt += " " + args.added_prompt # clean, extremely detailed, best quality, sharp, clean
            negative_prompt = args.negative_prompt #dirty, messy, low quality, frames, deformed, 
            print(validation_prompt)

            txt_path = image_name.replace("/LR/", "/lowlevel_prompt_q/").replace(".png", ".txt")
            fp = open(txt_path, "r")
            try:
                lowlevel_prompt = fp.readlines()[0].lstrip()
            except:
                lowlevel_prompt = ""

            #lowlevel_prompt += " " + "blurry, low quality, low resolution"
            ori_width, ori_height = validation_image.size
            resize_flag = False
            rscale = args.upscale

            validation_image = validation_image.resize((validation_image.size[0]*rscale, validation_image.size[1]*rscale))

            if min(validation_image.size) < args.process_size:
                validation_image = resize_preproc(validation_image)

            validation_image = validation_image.resize((validation_image.size[0]//8*8, validation_image.size[1]//8*8))
            resize_flag = True

            try:
                image = pipeline(
                        args, validation_prompt, validation_image, lowlevel_prompt = lowlevel_prompt, num_inference_steps=args.num_inference_steps, generator=generator, #height=height, width=width,
                        guidance_scale=args.guidance_scale, negative_prompt=negative_prompt, conditioning_scale=args.conditioning_scale,
                    ).images[0]
            except Exception as e:
                print(e)
                continue

            if True: #args.conditioning_scale < 1.0:
                image = adain_color_fix(image, validation_image)

            if resize_flag: 
                image = image.resize((ori_width*rscale, ori_height*rscale))

            folder_path, ext_path = os.path.split(image_name)
            num = args.xpsr_model_path.split("runs/")[-1]
            output_name = folder_path.replace("/LR", "/Prediction/" + num + "")
            os.makedirs(output_name, exist_ok=True)
            image.save(os.path.join(output_name, ext_path))


def metrics(args):
    dir = args.image_path
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(pyiqa.list_models())
    niqe_params_path = 'niqe/'
    #folders = os.listdir(dir)
    #folders = ["DIV2K_valid_old"]
    #folders = ["DIV2K_valid_old", "RealSR_CenterCrop", "DrealSR_CenterCrop"]
    #folders = [ "RealSR_CenterCrop"]
    folders = ["RealSR_CenterCrop", "DrealSR_CenterCrop"]
    prediction_dir = args.xpsr_model_path.split("runs")[-1]+ ""
    img_preproc = transforms.Compose([
        transforms.ToTensor(),
    ])

    psnr_metric = pyiqa.create_metric('psnr', device=device)
    ssim_metric = pyiqa.create_metric('ssim', device=device)
    fid_metric = pyiqa.create_metric('fid', device=device)
    maniqa_metric = pyiqa.create_metric('maniqa', device=device)
    lpips_iqa_metric = pyiqa.create_metric('lpips', device=device)
    clipiqa_iqa_metric = pyiqa.create_metric('clipiqa', device=device)
    musiq_iqa_metric = pyiqa.create_metric('musiq', device=device)
    dists_iqa_metric = pyiqa.create_metric('dists', device=device)
    niqe_iqa_metric = pyiqa.create_metric('niqe', device=device)

    print(prediction_dir)
    for folder in folders:
        print(folder)
        gt_img_paths = []

        psnr_folder = []
        ssim_folder = []
        lpips_score = []
        dists_score = []
        niqe_score = []
        lpips_iqa = []
        musiq_iqa = []
        maniqa_iqa = []
        clip_iqa = []
        gt_img_paths.extend(sorted(glob.glob(f'{dir}/{folder}/HR/*.png'))[:])
        real_image_folder = dir + "/" + folder + "/HR"
        generated_image_folder = real_image_folder.replace("/HR", "/Prediction" + prediction_dir)

        for gt_img_path in gt_img_paths:
            GT_image = img_preproc(Image.open(gt_img_path).convert('RGB'))
            prediction_img_path = gt_img_path.replace("/HR/", "/Prediction" + prediction_dir + "/")
            Prediction_img = img_preproc(Image.open(prediction_img_path).convert('RGB'))
            #psnr_folder.append(psnr_metric(prediction_img_path, gt_img_path))

            img1 = rgb2ycbcr_pt(img2tensor(io.imread(gt_img_path)),  y_only=True).to(torch.float64)
            img2 = rgb2ycbcr_pt(img2tensor(io.imread(prediction_img_path)),  y_only=True).to(torch.float64)
            img1 = torch.squeeze(img1)
            img2 = torch.squeeze(img2)
            #ssim_folder.append(ssim(np.array(img1 * 255.), np.array(img2 * 255.), data_range=255))
            
            ssim_folder.append(ssim_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            psnr_folder.append(psnr_metric(img1.unsqueeze(0).unsqueeze(0), img2.unsqueeze(0).unsqueeze(0)))
            lpips_iqa.append(lpips_iqa_metric(prediction_img_path, gt_img_path))
            clip_iqa.append(clipiqa_iqa_metric(prediction_img_path))
            musiq_iqa.append(musiq_iqa_metric(prediction_img_path))
            maniqa_iqa.append(maniqa_metric(prediction_img_path))
            dists_score.append(dists_iqa_metric(prediction_img_path, gt_img_path))
            niqe_score.append(niqe_iqa_metric(prediction_img_path))

        m_psnr = sum(psnr_folder) / len(psnr_folder)
        m_ssim = sum(ssim_folder) / len(ssim_folder)
        print(f"PSNR = {m_psnr}")
        print(f"SSIM = {m_ssim}")
        m_lpips = sum(lpips_iqa)/len(lpips_iqa)
        print(f"LPIPS = {m_lpips.item()}")
        m_dists = sum(dists_score)/len(dists_score)
        print(f"DISTS = {m_dists}")
        m_niqe = sum(niqe_score)/len(niqe_score)
        print(f"NIQE = {m_niqe}")
        clipiqa = sum(clip_iqa)/len(clip_iqa)
        print(f"CLIP-IQA = {clipiqa.item()}")
        musiq = sum(musiq_iqa)/len(musiq_iqa)
        print(f"MUSIQ = {musiq.item()}")
        maniqa = sum(maniqa_iqa)/len(maniqa_iqa)
        print(f"MANIQA = {maniqa.item()}")
        fid_value = fid_metric(real_image_folder, generated_image_folder)
        print(f"FID = {fid_value}")




if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--pretrained_model_path", type=str, default="checkpoints/stable-diffusion-v1-5")
    parser.add_argument("--xpsr_model_path", type=str, default="runs/xpsr_lightdataset_nodegradation/checkpoint-64000")
    parser.add_argument("--latest", type=bool, default=False)
    parser.add_argument("--personalized_model_path", type=str, default="majicmixRealistic_v6.safetensors") # toonyou_beta3.safetensors, majicmixRealistic_v6.safetensors, unet_disney
    parser.add_argument("--control_type", choices=['realisr', 'grayscale'], nargs='?', default="realisr")
    parser.add_argument('--high_level_info', choices=['classification', 'detection', 'caption'], nargs='?', default='')
    parser.add_argument("--prompt", type=str, default="")
    parser.add_argument("--added_prompt", type=str, default="clean, high-resolution, 8k")
    parser.add_argument("--negative_prompt", type=str, default="blurry, dotted, noise, raster lines, unclear, lowres, over-smoothed")
    parser.add_argument("--image_path", type=str, default="examples")
    parser.add_argument("--mixed_precision", type=str, default="fp16") # no/fp16/bf16
    parser.add_argument("--guidance_scale", type=float, default=7.5)
    parser.add_argument("--conditioning_scale", type=float, default=1.0)
    parser.add_argument("--blending_alpha", type=float, default=1.0)
    parser.add_argument("--multiplier", type=float, default=0.6)
    parser.add_argument("--num_inference_steps", type=int, default=50)
    parser.add_argument("--process_size", type=int, default=512) # 512?
    parser.add_argument("--decoder_tiled_size", type=int, default=224) # for 24G
    parser.add_argument("--encoder_tiled_size", type=int, default=1024) # for 24G
    parser.add_argument("--latent_tiled_size", type=int, default=320) # for 24G
    parser.add_argument("--latent_tiled_overlap", type=int, default=8) # for 24G
    parser.add_argument("--upscale", type=int, default=4)
    parser.add_argument("--use_personalized_model", action="store_true")
    parser.add_argument("--use_xpsr_light", action="store_true")
    parser.add_argument("--init_latent_with_noise", action="store_true")
    parser.add_argument("--added_noise_level", type=int, default=800)
    parser.add_argument("--offset_noise_scale", type=float, default=0.0)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()
    with open("configs/xpsr_test.yaml", 'r') as file:
        config_data = yaml.safe_load(file)
    for key, value in config_data.items():
        setattr(args, key, value)
    if args.latest:
        dirs = os.listdir(args.xpsr_model_path.split("/checkpoint")[0])
        dirs = [d for d in dirs if d.startswith("checkpoint")]
        dirs = sorted(dirs, key=lambda x: int(x.split("-")[1]))
        path = dirs[-1] if len(dirs) > 0 else None
        args.xpsr_model_path = args.xpsr_model_path.replace(args.xpsr_model_path.split("/")[-1], path)


    main(args)
    metrics(args)

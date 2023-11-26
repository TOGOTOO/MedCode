import os
import torch
import numpy as np
import SimpleITK as sitk
from skimage.metrics import structural_similarity
from tqdm import tqdm

#将in_path路径下数据读入一个列表中
def read_img(in_path):
    img_lit = []
    filenames = os.listdir(in_path)
    for f in tqdm(filenames):
        img = sitk.ReadImage(os.path.join(in_path, f))
        img_vol = sitk.GetArrayFromImage(img)
        img_lit = img_lit.append(img_vol)
    return img_lit

#将处理后的数据存入out_path中
def write_img(vol, out_path, ref_path, new_spacing=None):
    """
    参数分别表示为：
    vol:要存储的数据
    out_path:存储路径
    ref_path:参考图路径
    new_spacing:是否有新的采样间隔
    """
    img_ref = sitk.ReadImage(ref_path)
    img = sitk.GetImageFromArray(vol)
    img.SetDirection(img_ref.GetDirection())
    if new_spacing is None:
        img.SetSpacing(img_ref.GetSpacing())
    else:
        img.SetSpacing(tuple(new_spacing))
    img.SetOrigin(img_ref.GetOrigin())
    sitk.WriteImage(img, out_path)
    print('Save to:', out_path)

#归一化
def normal(in_img):
    value_max = np.max(in_img)
    value_min = np.min(in_img)
    return (in_img - value_min) / (value_max - value_min)

#psnr计算
def psnr(img, ground_turth):
    mse = np.mean((img - ground_turth) ** 2)
    if mse == 0:
        return float('inf')
    data_range = np.max(ground_turth) - np.min(ground_turth)
    return 20 * np.log10(data_range) - 10 * np.log10(mse)

#ssim计算
def ssim(image, ground_truth):
    data_range = np.max(ground_truth) - np.min(ground_truth)
    return structural_similarity(image, ground_truth, data_range=data_range)
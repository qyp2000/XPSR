import os
import glob
import torch
import random
import numpy as np
from PIL import Image
from functools import partial
from .degradations import random_add_gaussian_noise_pt, random_add_poisson_noise_pt, random_add_speckle_noise_pt, random_add_saltpepper_noise_pt, bivariate_Gaussian

from torch import nn
from torchvision import transforms
from torch.utils import data as data

from .realesrgan import RealESRGAN_degradation
from myutils.img_util import convert_image_to_fn
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
def exists(x):
    return x is not None

class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets_share_1/quyunpeng/trainset", 
                image_size=512,
                tokenizer=None,
                accelerator=None,
                control_type=None,
                null_text_ratio=0.5,
                original_image_ratio = 0.2,
                center_crop=False,
                random_flip=True,
                resize_bak=True,
                convert_image_to="RGB",
        ):
        super(LocalImageDataset, self).__init__()
        self.tokenizer = tokenizer
        self.control_type = control_type
        self.resize_bak = resize_bak
        self.null_text_ratio = null_text_ratio
        self.original_image_ratio = original_image_ratio

        self.degradation = RealESRGAN_degradation('dataloader/params_realesrgan.yml', device='cpu')

        maybe_convert_fn = partial(convert_image_to_fn, convert_image_to, image_size) if exists(convert_image_to) else nn.Identity()
        self.img_preproc = transforms.Compose([
            #transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])

        self.img_paths = []
        #folders = ['DIV2K_train_HR', 'Flickr2K_HR-1', 'DIV8K-3', 'DIV8K-2', 'Unsplash2K', 'Flickr2K_HR-0', 'DIV8K-5',  'DIV8K-1', 'DIV8K-0', 'DIV8K_2K', 'DIV8K-4', 'Flickr2K_HR-2']
        for index in range(0, 60):
            folder = "dataset_" + str(index)
            a = []
            b = []
            data_folders = os.listdir(os.path.join(pngtxt_dir, folder + "/HR"))
            for data_folder in data_folders:
                self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/HR/{data_folder}/*'))[:])
                a.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/HR/{data_folder}/*'))[:])
                b.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/lowlevel_prompt_q/{data_folder}/*'))[:])
            if len(a)-len(b) > 0:
                print(folder)
                print(len(a)-len(b))

        print(len(self.img_paths))#1153649

    def tokenize_caption(self, caption):            
        inputs = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length, padding="max_length", truncation=True, return_tensors="pt"
        )

        return inputs.input_ids

    def __getitem__(self, index):
        example = dict()

        # load image
        img_path = self.img_paths[index]
        GT_image_t = Image.open(img_path).convert('RGB')
        GT_image_t = self.img_preproc(GT_image_t)
        example["pixel_values"] = GT_image_t.squeeze(0) * 2.0 - 1.0

        img_path = self.img_paths[index].replace("/HR/", "/LR/")
        LR_image_t = Image.open(img_path).convert('RGB')
        LR_image_t = self.img_preproc(LR_image_t)
        example["conditioning_pixel_values"] = LR_image_t.squeeze(0)
        num = int(self.img_paths[index].split("/dataset_")[-1].split("/HR/")[0])

        if random.random() < self.null_text_ratio:
            caption = ""
            example["highlevel_prompt"] = self.tokenize_caption(caption).squeeze(0)
            example["lowlevel_prompt"] = self.tokenize_caption(caption).squeeze(0)

            if random.random() < self.original_image_ratio:
                example["conditioning_pixel_values"] = GT_image_t.squeeze(0) #random_add_gaussian_noise_pt(GT_image_t.unsqueeze(0),sigma_range=[1,7],clip=False,rounds=False,gray_prob=0,).squeeze(0)

        else:
            txt_path = self.img_paths[index].replace("/HR/", "/highlevel_prompt_GT/").replace(".png", ".txt")
            fp = open(txt_path, "r")
            try:
                high_caption = fp.readlines()[0].lstrip()
            except:
                try: 
                    txt_path = self.img_paths[index].replace("/HR/", "/highlevel_prompt/").replace(".png", ".txt")
                    fp = open(txt_path, "r")
                    high_caption = fp.readlines()[0].lstrip()
                except:
                    high_caption = ""
            if self.tokenizer is not None:
                example["highlevel_prompt"] = self.tokenize_caption(high_caption).squeeze(0)
            fp.close()


            txt_path = self.img_paths[index].replace("/HR/", "/lowlevel_prompt_q/").replace(".png", ".txt")
            fp = open(txt_path, "r")
            try:
                caption = fp.readlines()[0].lstrip()
            except:
                caption = ""
            if self.tokenizer is not None:
                example["lowlevel_prompt"] = self.tokenize_caption(caption).squeeze(0)
            fp.close()

            
            if random.random() < self.original_image_ratio:
                example["conditioning_pixel_values"] = GT_image_t.squeeze(0)#random_add_gaussian_noise_pt(GT_image_t.unsqueeze(0),sigma_range=[1,7],clip=False,rounds=False,gray_prob=0,).squeeze(0)
                txt_path = self.img_paths[index].replace("/HR/", "/lowlevel_prompt_q_GT/").replace(".png", ".txt")
                fp = open(txt_path, "r")
                try:
                    caption = fp.readlines()[0].lstrip()
                except:
                    caption = ""
                example["lowlevel_prompt"] = self.tokenize_caption(caption).squeeze(0)
            


        example["input_ids"] = self.tokenize_caption("").squeeze(0)
        return example

    def __len__(self):
        return len(self.img_paths)
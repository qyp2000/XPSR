import os
import sys
import glob
from transformers import AutoProcessor, LlavaForConditionalGeneration
sys.path.append(os.getcwd())
import cv2
from PIL import Image
from torch.utils import data as data
import torch
import torch.nn.functional as F
from pytorch_lightning import seed_everything
from tqdm import tqdm
import sys
import argparse
from torchvision import transforms
from transformers import AutoModel, AutoTokenizer
CUDA_LAUNCH_BLOCKING = 1

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='testset', help='the save path of the training dataset.')
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ') 
parser.add_argument("--end_gpu", type=int, default=1, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ')   
parser.add_argument("--batch_size", type=int, default=10, help='smaller batch size means much time but more extensive degradation for making the training dataset.')  
parser.add_argument("--epoch", type=int, default=8, help='decide how many epochs to create for the dataset.')
args = parser.parse_args()


from accelerate import Accelerator
accelerator = Accelerator()

torch.set_grad_enabled(False)

# init model and tokenizer
#model = AutoModel.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True).cuda().eval()
#tokenizer = AutoTokenizer.from_pretrained('DLight1551/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True)
model = AutoModel.from_pretrained('checkpoints/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True).to(accelerator.device).eval()
tokenizer = AutoTokenizer.from_pretrained('checkpoints/internlm-xcomposer-vl-7b-qinstruct-full', trust_remote_code=True)
model.tokenizer = tokenizer


class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets_share/quyunpeng/train_datasets/", 
                image_size=512,
                start = 0,
                end = 1,
        ):
        super(LocalImageDataset, self).__init__()
        self.img_paths = []
        self.img_preproc = transforms.Compose([
            #transforms.Lambda(maybe_convert_fn),
            #transforms.Resize(image_size),
            #transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            #transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ])
        
        for folder in ["DIV2K_valid_old", "RealSR_CenterCrop", "DrealSR_CenterCrop"]:#os.listdir(pngtxt_dir):
            self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/HR/*.png')))
            os.makedirs(f'{pngtxt_dir}/{folder}/lowlevel_prompt_q/', exist_ok=True)
            os.makedirs(f'{pngtxt_dir}/{folder}/lowlevel_prompt_q_GT/', exist_ok=True)

        print(len(self.img_paths))

    def __getitem__(self, index):
        example = dict()
        gt_save_path =  self.img_paths[index]
        lr_save_path = gt_save_path.replace("/HR/","/LR/")

        return gt_save_path, lr_save_path

    def __len__(self):
        return len(self.img_paths)


# Single-Turn Text-Image Dialogue
text = 'Describe and evaluate the quality of the image.'


train_dataset = LocalImageDataset(pngtxt_dir = args.save_dir, start = args.start_gpu, end = args.end_gpu)
train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False,
                )

train_dataset_loader = accelerator.prepare(train_dataset_loader
)


with torch.no_grad():
    for i, example in enumerate(tqdm(train_dataset_loader, disable=not accelerator.is_local_main_process)):
        # Generate
        gt_path, lr_path = example
        gt_path = gt_path[0]
        lr_path = lr_path[0]
        gt_save_path = gt_path.replace("/HR/","/lowlevel_prompt_q_GT/").replace(".png",".txt")
        lr_save_path = lr_path.replace("/LR/","/lowlevel_prompt_q/").replace(".png",".txt")

        if os.path.exists(gt_save_path) and  os.path.exists(lr_save_path):
            continue

        try:
            response = model.generate(text, gt_path)
            # 将预测结果保存到输出文件中
            with open(gt_save_path, 'w') as f:
                f.write(response.replace('\n', ' '))

            response = model.generate(text, lr_path)
            # 将预测结果保存到输出文件中
            with open(lr_save_path, 'w') as f:
                f.write(response.replace('\n', ' '))

        except:
            with open(os.path.dirname(gt_save_path) + "_abnormal.txt", 'a') as f:
                f.write(os.path.basename(gt_save_path)+"\n")
            
            with open(gt_save_path, 'w') as f:
                f.write('')
            with open(lr_save_path, 'w') as f:
                f.write('')
            print(gt_save_path)
            sys.exit(1)

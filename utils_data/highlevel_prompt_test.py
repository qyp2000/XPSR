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
CUDA_LAUNCH_BLOCKING = 1

parser = argparse.ArgumentParser()
parser.add_argument("--save_dir", type=str, default='testset', help='the save path of the training dataset.')
parser.add_argument("--start_gpu", type=int, default=0, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ') 
parser.add_argument("--end_gpu", type=int, default=10, help='if you have 5 GPUs, you can set it to 1/2/3/4/5 on five gpus for parallel processing., which will save your time. ')   
parser.add_argument("--batch_size", type=int, default=10, help='smaller batch size means much time but more extensive degradation for making the training dataset.')  
parser.add_argument("--epoch", type=int, default=8, help='decide how many epochs to create for the dataset.')
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(
    model_id, 
    torch_dtype=torch.float16, 
    low_cpu_mem_usage=True, 
)
processor = AutoProcessor.from_pretrained(model_id)
from accelerate import Accelerator
accelerator = Accelerator()


class LocalImageDataset(data.Dataset):
    def __init__(self, 
                pngtxt_dir="/datasets_share/quyunpeng/train_datasets/", 
                image_size=512,
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
        for folder in ["DIV2K_valid", "RealSR_CenterCrop", "DrealSR_CenterCrop"]:#os.listdir(pngtxt_dir):
            self.img_paths.extend(sorted(glob.glob(f'{pngtxt_dir}/{folder}/HR/*.png')))
            os.makedirs(f'{pngtxt_dir}/{folder}/highlevel_prompt/', exist_ok=True)
            os.makedirs(f'{pngtxt_dir}/{folder}/highlevel_prompt_GT/', exist_ok=True)

        self.processor = AutoProcessor.from_pretrained("llava-hf/llava-1.5-7b-hf")
        print(len(self.img_paths))

    def __getitem__(self, index):
        example = dict()
        gt_save_path =  self.img_paths[index]
        lr_save_path = gt_save_path.replace("/HR/","/LR/")
        highlevel_prompt_path = lr_save_path.replace("/LR/", "/highlevel_prompt/").replace(".png", ".txt")
        highlevel_prompt_gt_path = lr_save_path.replace("/LR/", "/highlevel_prompt_GT/").replace(".png", ".txt")
        lr_image_read = Image.open(lr_save_path).convert('RGB')
        hr_image_read = Image.open(gt_save_path).convert('RGB')
        highlevel_prompt = "<image>\nUSER: Please provide a descriptive summary of the content of this image. \nASSISTANT:"
        
        image_llava_input = [lr_image_read, hr_image_read]
        prompt_llava_input = [highlevel_prompt, highlevel_prompt]
        path = [highlevel_prompt_path, highlevel_prompt_gt_path]

        example["path"] = path
        example["prompt"] = prompt_llava_input
        example["image"] = image_llava_input
        inputs = processor(text=example["prompt"], images=example["image"], padding=True, return_tensors="pt")

        return inputs, example["path"]

    def __len__(self):
        return len(self.img_paths)


train_dataset = LocalImageDataset(pngtxt_dir = args.save_dir)
train_dataset_loader = torch.utils.data.DataLoader(
                    train_dataset, batch_size=1, num_workers=8, pin_memory=True, shuffle=False,
                )

model.to(accelerator.device)


model, train_dataset_loader = accelerator.prepare(
    model, train_dataset_loader
)

with torch.no_grad():
    for i, example in enumerate(tqdm(train_dataset_loader, disable=not accelerator.is_local_main_process)):
        # Generate
        inputs, paths = example
        inputs = inputs.to(accelerator.device)
        for key, value in inputs.items():
            inputs[key] = value.squeeze(0)

        #if os.path.exists(paths[0][0]) and  os.path.exists(paths[1][0])  and os.path.exists(paths[2][0]):
        #    continue
        if os.path.exists(paths[0][0]):
            continue

            
        try:
            generate_ids = model.module.generate(**inputs, max_length=200)
            generated_text = processor.batch_decode(generate_ids, skip_special_tokens=True)

            for index in range(len(generated_text)):
                text = generated_text[index].split("ASSISTANT:")[-1]
                # 将预测结果保存到输出文件中
                with open(paths[index][0], 'w') as f:
                    f.write(text.replace('\n', ' '))
            
        except:
            with open(os.path.dirname(os.path.dirname(paths[0][0])) + "_abnormal_new.txt", 'a') as f:
                f.write(os.path.basename(paths[0][0])+"\n")
            
            for index in range(len(paths)):
                # 将预测结果保存到输出文件中
                with open(paths[index][0], 'w') as f:
                    f.write('')
            print(paths[0][0])
            sys.exit(1)

        del inputs

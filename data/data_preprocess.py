from torch.utils.data import Dataset
from PIL import Image
import os
import torchvision.transforms.functional as TF
import torchvision.transforms as T


# LasHeR dataset class
class data_preprocess(Dataset):

    def __init__(self, root_dir):
        self.root_dir = root_dir
        img_path = os.listdir(self.root_dir)
        img_ir_list_path = [self.root_dir + '/' + x for x in img_path]
        self.img_ir_path = img_ir_list_path

    def transform(self, img_ir, img_vis):
        # grayscale
        img_ir = TF.to_grayscale(img_ir)
        img_vis = TF.to_grayscale(img_vis)
        # to tensor
        img_ir = TF.to_tensor(img_ir)
        img_vis = TF.to_tensor(img_vis)
        # Normalize [-1, 1]
        # img_ir = TF.normalize(img_ir, 0.5, 0.5)
        # img_vis = TF.normalize(img_vis, 0.5, 0.5)
        return img_ir, img_vis

    def __getitem__(self, idx):
        img_ir_name = self.img_ir_path[idx]
        img_vis_name = img_ir_name.replace('IR/', 'VIS/')
        img_ir = Image.open(img_ir_name)
        img_vis = Image.open(img_vis_name)
        # self-defined transform
        img_ir, img_vis = self.transform(img_ir, img_vis)
        return img_ir, img_vis

    def __len__(self):
        return len(self.img_ir_path)

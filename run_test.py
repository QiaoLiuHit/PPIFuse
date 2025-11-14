import os
import torch
from PIL import Image
import torchvision.transforms as T
from torchvision.utils import save_image
from models.backbone import FusionUNet
import statistics

# method name
method = 'PPIFuse'

# test dataset
dataset = 'TNO'

# set gpu id
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
# source image path
root_path = '/data/zhangqh/DataSet/Test_Dataset/TNO_test'
# load all images
img_list = os.listdir(os.path.join(root_path, 'ir'))
# test for multiple epoch's checkpoint or single checkpoint
window_size = 64

# path for save fused image
fused_path = './FusedImg/TNO' + '/'
# load model
model = FusionUNet.Net()
model_path = './checkpoints/PPIFuse_10' +'.pth'
checkpoint = torch.load(model_path,weights_only=True)
model.load_state_dict(checkpoint['net'])

# set for test
model.to(device)
model.eval()

# load all images

for img in img_list:

    img_ir_path = os.path.join(root_path, 'ir', img)
    img_vis_path = img_ir_path.replace('ir/', 'vi/')

    # read ir and vis images
    img_ir = Image.open(img_ir_path)
    img_vis = Image.open(img_vis_path)
    ori_size = img_ir.size

    # transform
    transform = T.Compose([T.Grayscale(), T.ToTensor()])
    img_ir = transform(img_ir)
    img_vis = transform(img_vis)

    img_ir = img_ir.view(1, 1, ori_size[1],ori_size[0]).to(device)
    img_vis = img_vis.view(1, 1, ori_size[1],ori_size[0]).to(device)
    # test
    with torch.no_grad():
        _, _, h_old, w_old = img_ir.size()
        if h_old % window_size != 0:
            h_pad = (h_old // window_size + 1) * window_size - h_old
        else:
            h_pad = 0
        if w_old % window_size != 0:
            w_pad = (w_old // window_size + 1) * window_size - w_old
        else:
            w_pad = 0
        img_ir = torch.cat([img_ir, torch.flip(img_ir, [2])], 2)[:, :, :h_old + h_pad, :]
        img_ir = torch.cat([img_ir, torch.flip(img_ir, [3])], 3)[:, :, :, :w_old + w_pad]
        img_vis = torch.cat([img_vis, torch.flip(img_vis, [2])], 2)[:, :, :h_old + h_pad, :]
        img_vis = torch.cat([img_vis, torch.flip(img_vis, [3])], 3)[:, :, :, :w_old + w_pad]
        out = model(img_ir, img_vis)
        out = out[..., :h_old, :w_old]
        max = out.max()
        min = out.min()
        out = (out - min)/(max - min)
        out = out.view(1, ori_size[1], ori_size[0])
        fusion_img = out
        if not os.path.exists(fused_path):
            os.mkdir(fused_path)
        save_image(fusion_img, fused_path + img)


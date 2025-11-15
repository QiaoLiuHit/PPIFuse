import time
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from data.data_preprocess import data_preprocess
from models.loss import loss_vif
from models.backbone import FusionUNet
from models.loss import energy_conservation_loss

bar_format = '{l_bar}{bar}| {n_fmt}/{total_fmt} '

# prepare training dataset, please config your dataset path
root_MSRS_train_dir = '/data/zhangqh/DataSet/new_MSRS/IR'
root_LLVIP_train_dir = '/data/zhangqh/DataSet/new_LLVIP/IR'

MSRS_train = data_preprocess(root_MSRS_train_dir)
LLVIP_train = data_preprocess(root_LLVIP_train_dir)

TraningDatset = MSRS_train + LLVIP_train
train_dataloader = DataLoader(TraningDatset, batch_size=16, drop_last=True, shuffle=True)

# Create PPIFuse network model
model = FusionUNet.Net()
device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
model = model.to(device)

# checkpoint = torch.load("./checkpoints/PPIFuse_10.pth")
# model.load_state_dict(checkpoint['net'])

# define loss
compute_loss = loss_vif.fusion_loss_vif()
energy_loss = energy_conservation_loss.energy_conservation_loss().to(device)


# define optimizer
learning_rate = 0.0001
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.8)

# training epoch
epoch = 10

for i in range(1, epoch+1):
    start_time = time.time()
    total_train_step = 0
    print("training on {} epoch starting".format(i))
    model.train()

    for idx, data in enumerate(tqdm(train_dataloader, bar_format=bar_format)):

        img_ir, img_vis = data
        img_ir = img_ir.to(device)
        img_vis = img_vis.to(device)
        fusion_out = model(img_ir, img_vis)
        fusion_out = fusion_out.to(device)

        # training fusion network
        # loss function
        energy_consistency_loss = energy_loss(fusion_out, img_ir, img_vis)
        total_loss, loss_gradient, loss_l1, loss_SSIM = compute_loss(img_ir, img_vis, fusion_out)
        total_loss = total_loss + energy_consistency_loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        # print training information
        total_train_step = total_train_step + 1
        if total_train_step % 10 == 0:
            print("training with {} epoch and {} times and total loss is {}, "
                  "loss_gradient is {}, loss_l1 is {}, loss_SSIM is {}, energy_consistency_loss is {}".
                  format(i, total_train_step,round(total_loss.item(), 3),round(loss_gradient.item(), 3),
                  round(loss_l1.item(), 3),round(loss_SSIM.item(), 3), round(energy_consistency_loss.item(),3)))



    end_time = time.time()  # 记录周期结束的时间
    time_consume = int(end_time - start_time) // 60
    print(f"Epoch {i + 1} finished in {time_consume} minute")
    scheduler.step()
    # save checkpoint
    checkpoints = {
        "net": model.state_dict()
    }
    torch.save(checkpoints, "./checkpoints/PPIFuse_{}.pth".format(i))

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch
import torch.nn as nn
import torch.optim as optim
from mednext.nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
import data_loader
import yaml
import argparse 
import os

'''
In this script, we provide a basic (and simple) pipeline designed for successful execution.
There are numerous advanced AI methodologies and strategies that could potentially improve the model's performance. 
We encourage participants to explore these AI technologies independently. The organizers will not provide much support for these explorations.
Please note that discussions/questions about AI tech explorations are not supposed to be raised in the repository issues.

Reminder: The information provided in the meta files is crucial, as it directly impacts how the reference is created. 
An example of how to use these information are provided in the data_loader.py. 
If you have questions related to clinical backgrounds, feel free to start a discussion.
'''

parser = argparse.ArgumentParser(description='Process some integers.')

parser.add_argument('cfig_path',  type = str)
parser.add_argument('--phase', default = 'train', type = str)
args = parser.parse_args()

cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ------------ data loader -----------------#
loaders = data_loader.GetLoader(cfig = cfig['loader_params'])
train_loader =loaders.train_dataloader()
# val_loader = loaders.val_dataloader()


# ------------- Network ------------------ #
model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
  num_classes = cfig['model_params']['out_channels'],
  model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
  kernel_size = cfig['model_params']['kernel_size'],   # 3x3x3 and 5x5x5 were tested in publication
  deep_supervision = cfig['model_params']['deep_supervision']
).to(device)

# model.load_state_dict(torch.load(cfig['pretrain_ckpt'], map_location=device), strict=False)
# from train_lightning import GDPLightningModel
# pl_module = GDPLightningModel.load_from_checkpoint('/data/result/GDP-HMM_Challenge/GDP-HMM_baseline/MedNeXtV1_InCh_8_OutCh_1_ModelID_B_KerSize_3_DeepSup_False_Lightning/best-train_loss=0.1841.ckpt', cfig=cfig, strict=True)
# model = pl_module.model.to(device)

# model = nn.DataParallel(model)


# ------------ loss -----------------------# 
optimizer = optim.Adam([{'params': model.parameters(), 'initial_lr':cfig['lr']}], lr=cfig['lr'])

# scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer,T_max= len(train_loader) * cfig['num_epochs'], last_epoch=cfig['num_epochs'])
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfig['num_epochs'])

criterion = nn.L1Loss()

# -----------Training loop --------------- #

nbatch_per_log = max(int(len(train_loader) / 20), 1)  

best_loss = 100
for epoch in range(cfig['num_epochs']):
    model.train()
    epoch_loss = 0
    for batch_idx, data_dict in enumerate(train_loader):
        # Forward pass
        outputs = model(data_dict['data'].float().to(device))
        # outputs = model(data_dict['prescribed_dose'].float().to(device), data_dict['data'].float().to(device))

        #if cfig['act_sig']:
        outputs = torch.sigmoid(outputs)
        outputs = outputs * cfig['scale_out']
        # outputs = outputs * 0.2 * torch.reshape(data_dict['prescribed_dose'][:, 0], (-1, 1, 1, 1, 1)).to(device)
        # print(outputs.size())
        # raise

        # loss = criterion(outputs, data_dict['label'].to(device))
        # loss = loss * cfig['scale_loss']

        label = data_dict['label'].float().to(device)
        loss1 = criterion(outputs.float(), label)
        ref_dose = label * 10.
        outputs = outputs * 10.
        body = data_dict['Body'].to(device).float()
        body = torch.where(body > 0.3, torch.ones_like(body), torch.zeros_like(body))
        th = 5
        isodose_5Gy_mask = ((ref_dose > th) | (outputs > th)) & (body > 0)
        isodose_ref_5Gy_mask = (ref_dose > th) & (body > 0)
        diff = ref_dose - outputs
        diff = torch.abs(diff)
        diff = torch.sum(diff * isodose_5Gy_mask.float(), dim=[1, 2, 3, 4])
        isodose_ref_5Gy_mask = torch.sum(isodose_ref_5Gy_mask, dim=[1, 2, 3, 4])
        loss2 = torch.mean(diff / isodose_ref_5Gy_mask.float())
        loss = loss1 * 1 + loss2 * 1

        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # scheduler.step()

        # epoch_loss += loss.item()
        epoch_loss += loss2.item()

        if batch_idx % nbatch_per_log == 0:
            current_lr = scheduler.get_last_lr()[0]
            print(f"Epoch [{epoch+1}/{cfig['num_epochs']}], Batch [{batch_idx+1}/{len(train_loader)}], LR: {current_lr:.6f}, Loss: {loss2.item():.4f}")


    # Average loss for the epoch
    avg_epoch_loss = epoch_loss / len(train_loader)
    print(f"Epoch [{epoch+1}/{cfig['num_epochs']}] Completed: Train Avg Loss: {avg_epoch_loss:.4f}")

    if avg_epoch_loss < best_loss:
        model_save_path = os.path.join(cfig['save_model_root'], 'best_model-epoch=' + str(epoch) + '-train_loss=' + str(avg_epoch_loss) + '.pth')
        torch.save(model.state_dict(), model_save_path)
        # torch.save(model.module.state_dict(), model_save_path)
        best_loss = avg_epoch_loss


























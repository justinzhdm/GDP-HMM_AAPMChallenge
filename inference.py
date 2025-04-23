import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2'
import torch
# from nnunet_mednext import create_mednext_v1
# from mednext.nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_v1
from mednext.nnunet_mednext.network_architecture.mednextv1.create_mednext_v1 import create_mednext_mi as create_mednext_v1
# from mednext.nnunet_mednext.network_architecture.mednextv1.create_mednext_v1_new import create_mednext_v1
import data_loader
import yaml
import argparse 
import os
import pdb
import numpy as np
import SimpleITK as sitk
import time
import copy
from scipy import ndimage
import torch.nn.functional as F

'''
--------------------------------- Attention !!! -----------------------------------------

This script is only provide the example how the inference can be run. 
Participants may need to modify the script or/and parameters to get resonable/good results (e.g., change the in_size, out_size etc.). 
The sample cases we used here are actually from the train or valid split. 
For the challenge, the train/validation/test splits are mutual excluded. The final ranking should be run on the test split.

'''



def offset_spatial_crop(roi_center=None, roi_size=None):
    """
    for crop spatial regions of the data based on the specified `roi_center` and `roi_size`.
    
    get the start and end of the crop
    
    Parameters:
        roi_center (tuple of int, optional): The center point of the region of interest (ROI).
        roi_size (tuple of int, optional): The size of the ROI in each spatial dimension.
        
    Returns:
        start & end: start and end offsets
    """
    
    if roi_center is None or roi_size is None:
        raise ValueError("Both `roi_center` and `roi_size` must be specified.")
    
    roi_center = [int(round(c)) for c in roi_center]
    roi_size = [int(round(s)) for s in roi_size]
    
    start = []
    end = []
    
    for i, (center, size) in enumerate(zip(roi_center, roi_size)):
        
        half_size = size // 2 # int(round(size / 2))
        start_i = max(center - half_size, 0)  # Ensure we don't go below 0
        end_i = max(start_i + size, start_i)
        #end_i = min(center + half_size + (size % 2), ori_size[i])  
        start.append(start_i)
        end.append(end_i)

    return start, end

def cropped2ori(crop_data, ori_size, isocenter, trans_in_size):

    '''
    crop_data: the cropped data
    ori_size: the original size of the data
    isocenter: the isocenter of the original data
    trans_in_size: the in_size parameter in the transfromation of loader
    '''

    assert (np.array(trans_in_size) == np.array(crop_data.shape)).all()

    start_coords, end_coords = offset_spatial_crop(roi_center = isocenter, roi_size = trans_in_size)

    # remove the padding
    crop_start, crop_end = [], []
    for i in range(len(ori_size)):
        if end_coords[i] > ori_size[i]:
            diff = end_coords[i] - ori_size[i]
            crop_start.append(diff // 2)
            crop_end.append(crop_data.shape[i] - diff + diff // 2)
        else:
            crop_start.append(0)
            crop_end.append(crop_data.shape[i])

    
    crop_data = crop_data[crop_start[0]: crop_end[0], crop_start[1]: crop_end[1], crop_start[2]: crop_end[2]]

    pad_out = np.zeros(ori_size)

    pad_out[start_coords[0]: end_coords[0], start_coords[1]: end_coords[1], start_coords[2]: end_coords[2]] = crop_data 
    
    return pad_out


def recover_resize(data, ori_size):
    # shape = data.size()[2:]
    # data = F.interpolate(data, scale_factor=(float(ori_size[0]) / float(shape[0]),
    #                                          float(ori_size[1]) / float(shape[1]),
    #                                          float(ori_size[2]) / float(shape[2])), mode="trilinear", align_corners=False)
    ori_size = [int(ori_size[0]), int(ori_size[1]), int(ori_size[2])]
    data = F.interpolate(data, size=ori_size, mode="trilinear", align_corners=False)
    return data

# def recover(data, bbox, padding):
#     data = copy.deepcopy(data)
#     if padding[0] > 0:
#         data = data[padding[0]:, :, :]
#     if padding[1] > 0:
#         data = data[:-padding[1], :, :]
#     if padding[2] > 0:
#         data = data[:, padding[2]:, :]
#     if padding[3] > 0:
#         data = data[:, :-padding[3], :]
#     if padding[4] > 0:
#         data = data[:, :, padding[4]:]
#     if padding[5] > 0:
#         data = data[:, :, :-padding[5]]
#     data = np.pad(data, ((bbox[0], bbox[1]), (bbox[2], bbox[3]), (bbox[4], bbox[5])), 'constant', constant_values=0)
#     return data

def recover_pad(data, padding, ori_size):
    data = copy.deepcopy(data)
    if padding[0] > 0:
        data = data[:, :, padding[0]:, :, :]
    if padding[1] > 0:
        data = data[:, :, :-padding[1], :, :]
    if padding[2] > 0:
        data = data[:, :, :, padding[2]:, :]
    if padding[3] > 0:
        data = data[:, :, :, :-padding[3], :]
    if padding[4] > 0:
        data = data[:, :, :, :, padding[4]:]
    if padding[5] > 0:
        data = data[:, :, :, :, :-padding[5]]
    ori_size = [int(ori_size[0]), int(ori_size[1]), int(ori_size[2])]
    assert data.size()[2] == ori_size[0] and data.size()[3] == ori_size[1] and data.size()[4] == ori_size[2]
    return data

if __name__ == "__main__": 

    parser = argparse.ArgumentParser(description='Process some integers.')

    parser.add_argument('cfig_path',  type = str)
    parser.add_argument('--phase', default = 'test', type = str)
    args = parser.parse_args()

    cfig = yaml.load(open(args.cfig_path), Loader=yaml.FullLoader)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ------------ data loader -----------------#
    loaders = data_loader.GetLoader(cfig = cfig['loader_params'])
    test_loader = loaders.test_dataloader()

    # temp_cfig = copy.deepcopy(cfig)
    # # temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data.csv'
    # temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data_infer_dummy.csv'
    # # temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data_infer_temp.csv'
    # loaders = data_loader.GetLoader(cfig=temp_cfig['loader_params'])
    # train_loader = loaders.train_dataloader_test()
    # val_loader = loaders.val_dataloader()

    # if cfig['model_from_lightning']:  # when the model is trained with training_lightning.py
    #     from train_lightning import GDPLightningModel
    #     pl_module = GDPLightningModel.load_from_checkpoint(cfig['save_model_path'], cfig = cfig, strict = True)
    #     model = pl_module.model.to(device)
    # else:      # when the model is trained with train.py
    model = create_mednext_v1(num_input_channels=cfig['model_params']['num_input_channels'],
                              num_classes=cfig['model_params']['out_channels'],
                              model_id=cfig['model_params']['model_id'],  # S, B, M and L are valid model ids
                              kernel_size=cfig['model_params']['kernel_size'],
                              deep_supervision=cfig['model_params']['deep_supervision']
                              ).to(device)
    # load pretrained model
    model.load_state_dict(torch.load(cfig['save_model_path'], map_location=device))
    print('load model ', cfig['save_model_path'])


    os.makedirs(cfig['save_pred_path'], exist_ok=True)

    # a=time.time()
    avg_loss = 0
    with torch.no_grad():
        model.eval()
        for batch_idx, data_dict in enumerate(test_loader):
        # for batch_idx, data_dict in enumerate(train_loader):
        # for batch_idx, data_dict in enumerate(val_loader):
            a = time.time()
            outputs = model(data_dict['data'].to(device))
            print('Time taken for average forward pass: ', time.time() - a, data_dict['data_path'][0])

            # if cfig['act_sig']:
            # outputs = torch.sigmoid(outputs)
            # outputs = outputs * cfig['scale_out']
            # outputs = outputs * 0.2 * torch.reshape(data_dict['prescribed_dose'][:, 0], (-1, 1, 1, 1, 1)).to(device)


            if cfig['loader_params']['in_size'] != cfig['loader_params']['out_size']:
                outputs = torch.nn.functional.interpolate(outputs, size = cfig['loader_params']['in_size'], mode = 'area')
            for index in range(len(outputs)):
                pad_out = np.zeros(data_dict['ori_img_size'][index].numpy().tolist())
                crop_data = outputs[index][0].cpu().numpy()
                ori_size = data_dict['ori_img_size'][index].numpy().tolist()
                isocenter = data_dict['ori_isocenter'][index].numpy().tolist()
                trans_in_size = cfig['loader_params']['in_size']


                # pred2orisize = cropped2ori(crop_data, ori_size, isocenter, trans_in_size) * cfig['loader_params']['dose_div_factor']
                # print(data_dict['bbox'][index].numpy().tolist())
                # print(data_dict['padding'][index].numpy().tolist())
                # pred2orisize = recover(crop_data, data_dict['bbox'][index].numpy().tolist(), data_dict['padding'][index].numpy().tolist())

                # pred2orisize = recover_resize(outputs[index:index+1,:,:,:,:], ori_size).cpu().numpy() * cfig['loader_params']['dose_div_factor']
                # ref_dose = recover(data_dict['label'][index:index + 1, :, :, :, :], ori_size).cpu().numpy() * cfig['loader_params']['dose_div_factor']
                # print(ref_dose.shape)
                pred2orisize = recover_pad(outputs[index:index + 1, :, :, :, :], data_dict['padding'][index].numpy().tolist(), ori_size).cpu().numpy() * cfig['loader_params']['dose_div_factor']


                np.save(os.path.join(cfig['save_pred_path'], data_dict['id'][index] + '_pred.npy'), pred2orisize)

                # ref_dose = data_dict['origin_label'].cpu().numpy() * cfig['loader_params']['dose_div_factor']
                # data_dict['Body'] = data_dict['origin_body'].cpu().numpy()
                # temp_outputs = pred2orisize
                # # print(np.max(pred2orisize), np.max(ref_dose))
                # th = 5
                # isodose_5Gy_mask = ((ref_dose > th) | (temp_outputs > th)) & (data_dict['Body'] > 0)
                # isodose_ref_5Gy_mask = (ref_dose > th) & (data_dict['Body'] > 0)
                # diff = ref_dose - temp_outputs
                # loss= np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
                # print(loss, data_dict['data_path'][0])
                # avg_loss += loss



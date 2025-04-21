import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
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

    temp_cfig = copy.deepcopy(cfig)
    # temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data.csv'
    temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data_infer_dummy.csv'
    # temp_cfig['loader_params']['csv_root'] = 'meta_files/meta_data_infer_temp.csv'
    loaders = data_loader.GetLoader(cfig=temp_cfig['loader_params'])
    train_loader = loaders.train_dataloader_test()
    val_loader = loaders.val_dataloader()

    if cfig['model_from_lightning']:  # when the model is trained with training_lightning.py
        from train_lightning import GDPLightningModel
        pl_module = GDPLightningModel.load_from_checkpoint(cfig['save_model_path'], cfig = cfig, strict = True)
        model = pl_module.model.to(device)
    else:      # when the model is trained with train.py
        model = create_mednext_v1( num_input_channels = cfig['model_params']['num_input_channels'],
        num_classes = cfig['model_params']['out_channels'],
        model_id = cfig['model_params']['model_id'],          # S, B, M and L are valid model ids
        kernel_size = cfig['model_params']['kernel_size'],   
        deep_supervision = cfig['model_params']['deep_supervision']   
        ).to(device)
        # load pretrained model 
        model.load_state_dict(torch.load(cfig['save_model_path'], map_location = device))
        print('load model ', cfig['save_model_path'])

    print('che', cfig['save_model_path'])

    # a=time.time()
    avg_loss = 0
    with torch.no_grad():
        model.eval()
        # for batch_idx, data_dict in enumerate(test_loader):
        # for batch_idx, data_dict in enumerate(train_loader):
        for batch_idx, data_dict in enumerate(val_loader):
            a = time.time()
            # Forward pass

            # if 'HN-HMR-009+S29Ag+MOS_41448' not in data_dict['data_path'][0]:
            # # if batch_idx <120:
            # #     print(data_dict['data_path'][0])
            #     continue


            outputs = model(data_dict['data'].to(device))
            # outputs = model(data_dict['prescribed_dose'].float().to(device), data_dict['data'].float().to(device))
            print('Time taken for average forward pass: ', time.time() - a, data_dict['data_path'][0])

            # if cfig['act_sig']:
            # outputs = torch.sigmoid(outputs)
            # outputs = outputs * cfig['scale_out']
            # outputs = outputs * 0.2 * torch.reshape(data_dict['prescribed_dose'][:, 0], (-1, 1, 1, 1, 1)).to(device)

            # # if 'HNSCC-01-0012+A4Ac+MOS_13679' in data_dict['data_path'][0]:
            # if 1:
            #     print(data_dict['data_path'][0])
            #     imgSITK = sitk.GetImageFromArray(outputs.cpu().numpy()[0, 0, :, :, :])
            #     sitk.WriteImage(imgSITK, "./output.mhd")
            #     imgSITK = sitk.GetImageFromArray(data_dict['data'].cpu().numpy()[0, 0, :, :, :])
            #     sitk.WriteImage(imgSITK, "./input.mhd")
            #     raise

            # if 'label' in data_dict.keys():
            #     # print ('L1 error is ', torch.nn.L1Loss()(outputs, data_dict['label'].to(device)).item())
            #     # loss = torch.nn.L1Loss()(outputs, data_dict['label'].to(device)).item() * cfig['scale_loss']
            #     ref_dose = data_dict['label'].cpu().numpy() * cfig['loader_params']['dose_div_factor']
            #     # data_dict['Body'] = data_dict['Body'].cpu().numpy()
            #     data_dict['Body'] = torch.where(data_dict['Body'] > 0.3, torch.ones_like(data_dict['Body']), torch.zeros_like(data_dict['Body'])).cpu().numpy()
            #     temp_outputs = outputs.cpu().numpy() * cfig['loader_params']['dose_div_factor']
            #     th = 5
            #     isodose_5Gy_mask = ((ref_dose > th) | (temp_outputs > th)) & (data_dict['Body'] > 0)
            #     isodose_ref_5Gy_mask = (ref_dose > th) & (data_dict['Body'] > 0)
            #     diff = ref_dose - temp_outputs
            #     # loss= np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
            #     diff = np.abs(diff)
            #     diff = np.sum(diff * isodose_5Gy_mask.astype(np.float32), axis=(1, 2, 3, 4))
            #     isodose_ref_5Gy_mask = np.sum(isodose_ref_5Gy_mask, axis=(1, 2, 3, 4))
            #     loss = np.mean(diff / isodose_ref_5Gy_mask.astype(np.float32))
            #     print(loss)
            #     avg_loss += loss



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

                # print(pred2orisize.shape, ori_size)
                # np.save(os.path.join(cfig['save_pred_path'], data_dict['id'][index] + '_pred.npy'), pred2orisize)

                ref_dose = data_dict['origin_label'].cpu().numpy() * cfig['loader_params']['dose_div_factor']
                data_dict['Body'] = data_dict['origin_body'].cpu().numpy()
                temp_outputs = pred2orisize
                # print(np.max(pred2orisize), np.max(ref_dose))
                th = 5
                isodose_5Gy_mask = ((ref_dose > th) | (temp_outputs > th)) & (data_dict['Body'] > 0)
                isodose_ref_5Gy_mask = (ref_dose > th) & (data_dict['Body'] > 0)
                diff = ref_dose - temp_outputs
                loss= np.sum(np.abs(diff)[isodose_5Gy_mask > 0]) / np.sum(isodose_ref_5Gy_mask)
                print(loss, data_dict['data_path'][0])
                avg_loss += loss


                # # # if loss > 4:
                # # if '0522c0001+9Ag+MOS_23629' in data_dict['data_path'][0]:
                # imgSITK = sitk.GetImageFromArray(temp_outputs[0, 0, :, :, :])
                # sitk.WriteImage(imgSITK, "./outputs.mhd")
                # imgSITK = sitk.GetImageFromArray(ref_dose[0, 0, :, :, :])
                # sitk.WriteImage(imgSITK, "./label.mhd")
                # raise

                # #
                # #     ref_dose = data_dict['origin_label'].cpu().numpy() * cfig['loader_params']['dose_div_factor']
                # #     imgSITK = sitk.GetImageFromArray(ref_dose[0, 0, :, :, :])
                # #     sitk.WriteImage(imgSITK, "./label.mhd")
                # #     data = F.interpolate(data_dict['origin_label'].float() * cfig['loader_params']['dose_div_factor'], size=(128, 96, 144), mode="trilinear", align_corners=False)
                # #     ref_dose = F.interpolate(data, size=(124, 103, 169), mode="trilinear", align_corners=False)
                # #     ref_dose = ref_dose.cpu().numpy()
                # #     imgSITK = sitk.GetImageFromArray(ref_dose[0, 0, :, :, :])
                # #     sitk.WriteImage(imgSITK, "./label2.mhd")
                #     body = data_dict['origin_body'].cpu().numpy()
                #     imgSITK = sitk.GetImageFromArray(body[0, 0, :, :, :])
                #     sitk.WriteImage(imgSITK, "./body.mhd")
                #     body = F.interpolate(data_dict['origin_body'].float() * cfig['loader_params']['dose_div_factor'], size=(128, 96, 144), mode="trilinear", align_corners=False)
                #     body = torch.where(body>0.8, torch.ones_like(body), torch.zeros_like(body)).bool()
                #     body = F.interpolate(body.float(), size=(124, 103, 169), mode="trilinear", align_corners=False)
                #     body = torch.where(body > 0.8, torch.ones_like(body), torch.zeros_like(body))
                #     body = body.cpu().numpy()
                #     imgSITK = sitk.GetImageFromArray(body[0, 0, :, :, :])
                #     sitk.WriteImage(imgSITK, "./body2.mhd")
                #     raise


            # print(pred2orisize.shape)
            # print('time', time.time() - a)

        # avg_loss = avg_loss / len(train_loader)
        # print(len(train_loader), avg_loss)
        # avg_loss = avg_loss / len(val_loader)
        # print(len(val_loader), avg_loss)

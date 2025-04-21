import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import torch



if __name__ == "__main__":

    state_dict = {}

    temp_dict = {}
    state_dict1 = torch.load('/home/zhudeming/GDP-HMM_AAPMChallenge/pretrainmodel_test/best_model-epoch=50-train_loss=1.0081536694760724.pth')
    for key in state_dict1.keys():
        temp_dict['m1.' + key] = state_dict1[key]
    state_dict.update(temp_dict)
    temp_dict = {}
    state_dict2 = torch.load('/home/zhudeming/GDP-HMM_AAPMChallenge/pretrainmodel_test/best_model-epoch=69-train_loss=0.9495604410822138.pth')
    for key in state_dict2.keys():
        temp_dict['m2.' + key] = state_dict2[key]
    state_dict.update(temp_dict)
    temp_dict = {}
    state_dict3 = torch.load('/home/zhudeming/GDP-HMM_AAPMChallenge/pretrainmodel_test/best_model-epoch=92-train_loss=0.9059736057843044.pth')
    for key in state_dict3.keys():
        temp_dict['m3.' + key] = state_dict3[key]
    state_dict.update(temp_dict)
    temp_dict = {}
    state_dict4 = torch.load('/home/zhudeming/GDP-HMM_AAPMChallenge/pretrainmodel_test/best_model-epoch=129-train_loss=0.8454008709925872.pth')
    for key in state_dict4.keys():
        temp_dict['m4.' + key] = state_dict4[key]
    state_dict.update(temp_dict)


    torch.save(state_dict, '/home/zhudeming/GDP-HMM_AAPMChallenge/pretrainmodel_test/test.pth')




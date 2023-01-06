import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
from torch.autograd import Variable
import torchvision
from torchvision import datasets
from torchvision import transforms, models
from torch.utils.data import Dataset
from PIL import Image
import os
import h5py
import numpy as np
import random
import cv2
import videotransforms
from pytorch_i3d import InceptionI3d



class VideoDataset(Dataset):
    def __init__(self, videos_dir, video_names, score, video_format='RGB', width=None, height=None):

        super(VideoDataset, self).__init__()
        self.videos_dir = videos_dir
        self.video_names = video_names
        self.score = score
        self.format = video_format
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.video_names)

    def __getitem__(self, idx):
        # video_name = self.video_names[idx] + '.avi'
        video_name = self.video_names[idx] 
        frames_list=[]
        assert self.format == 'YUV420' or self.format == 'RGB'
        if self.format == 'YUV420':
            video_data = cv2.VideoCapture(os.path.join(videos_dir, video_name), (height, width),inputdict={'-pix_fmt':'yuvj420p'})
            while True:
                success, frame = video_data.read()
                if not success:
                    break
                frames_list.append(frame)
        else:
            video_data = cv2.VideoCapture(os.path.join(videos_dir, video_name))
            while True:
                success, frame = video_data.read()
                if not success:
                    break
                frames_list.append(frame)
            video_data.release()
        video_score = self.score[idx]

        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.CenterCrop(224),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

        video_length = len(frames_list)
        video_channel = frames_list[0].shape[2]
        # video_height = frames_list[0].shape[0]
        # video_width = frames_list[0].shape[1]
        transformed_video = torch.zeros([video_length, video_channel,  224, 224])
        for frame_idx in range(video_length):
            frame = frames_list[frame_idx]
            frame = Image.fromarray(frame)
            frame = transform(frame)
            transformed_video[frame_idx] = frame

        sample = {'video': transformed_video,
                  'score': video_score}

        return sample


class ResNet50(nn.Module):
    def __init__(self):
        super(ResNet50, self).__init__()
        self.features = nn.Sequential(*list(models.resnet50(pretrained=True).children())[:-2])
        for p in self.features.parameters():
            p.requires_grad = False

    def forward(self, x):
        # features@: 7->res5c take output from 7th layer
        for ii, model in enumerate(self.features):
            x = model(x)
            if ii == 7:
                features_mean = nn.functional.adaptive_avg_pool2d(x, 1)
                features_std = global_std_pool2d(x)
                return features_mean, features_std


def global_std_pool2d(x):
    return torch.std(x.view(x.size()[0], x.size()[1], -1, 1),
                     dim=2, keepdim=True)


def get_features(video_data, frame_batch_size=16, device='cuda'):
    """feature extraction"""
    extractor = ResNet50().to(device)
    video_length = video_data.shape[0]
    frame_start = 0
    frame_end = frame_start + frame_batch_size
    output1 = torch.Tensor().to(device)
    extractor.eval()
    with torch.no_grad():
	    while frame_end < video_length:
	        batch = video_data[frame_start:frame_end].to(device)
	        features_mean, features_std = extractor(batch)
	        output1 = torch.cat((output1, features_mean), 0)
	        frame_end += frame_batch_size
	        frame_start += frame_batch_size

	    last_batch = video_data[frame_start:video_length].to(device)
	    features_mean, features_std = extractor(last_batch)
	    output = torch.cat((output1, features_mean), 0).squeeze()
    return output



videos_dir = '/media/user/Research EMB_Lab/CVD2014'  # videos dir
features_dir = 'CNN_I3D_ResNet_features_CVD2014_database/'  # features dir
datainfo = '/home/user/Desktop/pytorch-i3d-master/CVD2014_database.mat'  # database info: video_names, scores; video format, width, height, index, ref_ids, max_len, etc.
    

if not os.path.exists(features_dir):
    os.makedirs(features_dir)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  

Info = h5py.File(datainfo, 'r')
video_names = [Info[Info['video_names'][0, :][i]][()].tobytes()[::2].decode() for i in range(len(Info['video_names'][0, :]))]
scores = Info['scores'][0, :]
video_format = Info['video_format'][()].tobytes()[::2].decode()
# video_format = 'RGB'
width = int(Info['width'][0])
height = int(Info['height'][0])
dataset = VideoDataset(videos_dir, video_names, scores, video_format, width, height)
frame_batch_size=4

load_model='/home/user/Desktop/pytorch-i3d-master/models/rgb_imagenet.pt'

i3d = InceptionI3d(400, in_channels=3)
i3d.load_state_dict(torch.load(load_model))
# i3d.to(device)


i3d.train(False)

for i in range(len(dataset)):
    current_data = dataset[i]
    current_video = current_data['video']
    current_score = current_data['score']
    
    print('Video {}: length {}'.format(i, current_video.shape[0]))
    
    features = get_features(current_video, frame_batch_size, device)
    batch_frame = 8
    start_frame = 0
    features_resnet = torch.zeros([features.shape[0]//8, 2048],dtype = float)
    for j in range(features.shape[0] // 8):
        features_combined = torch.mean(features[start_frame:start_frame + batch_frame,:],0)
        features_resnet[j,:] = features_combined
        start_frame += batch_frame
    #--------------------------------------------------------------------------------
    current_video_i3d = current_data['video']
    current_video_i3d = torch.unsqueeze(current_video_i3d,0)
    current_video_i3d = current_video_i3d.permute(0, 2, 1, 3, 4)
    
    current_video_i3d = Variable(current_video_i3d)
    features_i3d = i3d.extract_features(current_video_i3d)
    features_i3d = features_i3d.squeeze()
    features_i3d = features_i3d.permute(1,0)
      
    if features_resnet.size(0)>features_i3d.size(0):
      output_features = torch.cat((features_resnet[0:features_i3d.size(0),:], features_i3d ), 1)
    else:
      output_features = torch.cat((features_resnet, features_i3d[0:features_resnet.size(0),:] ), 1) 
        
    np.save(features_dir + str(i) + '_resnet-50_res5c_i3d', output_features.to('cpu').detach().numpy())
    np.save(features_dir + str(i) + '_score', current_score)
    #------------------------------------------------------------------------
    

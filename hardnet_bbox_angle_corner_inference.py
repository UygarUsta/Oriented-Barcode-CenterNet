import cv2
import numpy as np
import os
import pandas as pd
import re
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import math
from sklearn.model_selection import train_test_split
import schedulefree
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
#from torchinfo import summary
import torch.optim as optim
from functools import partial
assert torch.cuda.is_available()

# Not always necessary depending on your hardware/GPU
#os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"





# Here we dont assume that our images are 512x512 we prefer to use HD images (more common)
input_width = 512
input_height = 512

# Model scale is 16, meaning that in the model prediction, we have heatmaps of dimensions 80 x 45
MODEL_SCALE = 4
workers = 8
# Batch size for training --> if your hardware supports it, try to increase this value
batch_size = 8


def select(hm, threshold):
    """
    Keep only local maxima (kind of NMS).
    We make sure to have no adjacent detection in the heatmap.
    """

    pred = hm > threshold
    pred_centers = np.argwhere(pred)

    for i, ci in enumerate(pred_centers):
        for j in range(i + 1, len(pred_centers)):
            cj = pred_centers[j]
            if np.linalg.norm(ci - cj) <= 2:
                score_i = hm[ci[0], ci[1]]
                score_j = hm[cj[0], cj[1]]
                if score_i > score_j:
                    hm[cj[0], cj[1]] = 0
                else:
                    hm[ci[0], ci[1]] = 0

    return hm

def pred2box(hm, offset, regr, cos_sin_hm, thresh=0.99):
    # make binding box from heatmaps
    # thresh: threshold for logits.
        
    # get center
    pred = hm > thresh
    pred_center = np.where(hm>thresh)
    
    # get regressions
    pred_r = regr[:,pred].T
    pred_angles = cos_sin_hm[:, pred].T
    
    #print("pred_angle", pred_angle)

    # wrap as boxes
    # [xmin, ymin, width, height]
    # size as original image.
    boxes = []
    scores = hm[pred]
    
    pred_center = np.asarray(pred_center).T
    #print(pred_r.shape)
    #print(pred_angles)
    #print(pred_angles.shape)
    
    for (center, b, pred_angle) in zip(pred_center, pred_r, pred_angles):
        #print(b)
        offset_xy = offset[:, center[0], center[1]]
        angle = np.arctan2(pred_angle[1], pred_angle[0])
        arr = np.array([(center[1]+offset_xy[0])*MODEL_SCALE, (center[0]+offset_xy[1])*MODEL_SCALE, 
                        b[0]*MODEL_SCALE, b[1]*MODEL_SCALE, angle])
        # Clip values between 0 and input_size
        #arr = np.clip(arr, 0, input_size)
        #print("Pred angle", i, pred_angle[i])
        # filter 
        #if arr[0]<0 or arr[1]<0 or arr[0]>input_size or arr[1]>input_size:
            #pass
        boxes.append(arr)
    return np.asarray(boxes), scores



# functions for plotting results
def showbox(img, hm, offset, regr, cos_sin_hm, thresh=0.9):
    boxes, _ = pred2box(hm, offset, regr, cos_sin_hm, thresh=thresh)
    
    sample = img

    for box in boxes:
        center = [int(box[0]), int(box[1])]
        cos_angle = np.cos(box[4])
        sin_angle = np.sin(box[4])
        rot = np.array([[cos_angle, sin_angle], [-sin_angle, cos_angle]])
        
        bottom_right = np.dot(rot, np.array([box[2]/2, box[3]/2]).reshape(2, 1)).reshape(2)
        top_right = np.dot(rot, np.array([box[2]/2, -box[3]/2]).reshape(2, 1)).reshape(2)
        top_left = np.dot(rot, np.array([-box[2]/2, -box[3]/2]).reshape(2, 1)).reshape(2)
        bottom_left = np.dot(rot, np.array([-box[2]/2, box[3]/2]).reshape(2, 1)).reshape(2)
        
        thickness = 3
        cv2.line(sample, (int(center[0]+bottom_right[0]), int(center[1]+bottom_right[1])),
                      (int(center[0]+top_right[0]), int(center[1]+top_right[1])),
                      (0, 220, 0), thickness)
        cv2.line(sample, (int(center[0]+bottom_right[0]), int(center[1]+bottom_right[1])),
                      (int(center[0]+bottom_left[0]), int(center[1]+bottom_left[1])),
                      (220, 220, 0), thickness)
        cv2.line(sample, (int(center[0]+top_left[0]), int(center[1]+top_left[1])),
                      (int(center[0]+bottom_left[0]), int(center[1]+bottom_left[1])),
                      (220, 220, 0), thickness)
        cv2.line(sample, (int(center[0]+top_left[0]), int(center[1]+top_left[1])),
                      (int(center[0]+top_right[0]), int(center[1]+top_right[1])),
                      (220, 220, 0), thickness)
    return sample


def load_model(model, model_path, optimizer=None, resume=False, 
               lr=None, lr_step=None):
  start_epoch = 0
  checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
  print('loaded {}, epoch {}'.format(model_path, checkpoint['epoch']))
  state_dict_ = checkpoint['state_dict']
  state_dict = {}
  
  # convert data_parallal to model
  for k in state_dict_:
    if k.startswith('module') and not k.startswith('module_list'):
      state_dict[k[7:]] = state_dict_[k]
    else:
      state_dict[k] = state_dict_[k]
  model_state_dict = model.state_dict()

  # check loaded parameters and created model parameters
  msg = 'If you see this, your model does not fully load the ' + \
        'pre-trained weight. Please make sure ' + \
        'you have correctly specified --arch xxx ' + \
        'or set the correct --num_classes for your own dataset.'
  for k in state_dict:
    if k in model_state_dict:
      if state_dict[k].shape != model_state_dict[k].shape:
        print('Skip loading parameter {}, required shape{}, '\
              'loaded shape{}. {}'.format(
          k, model_state_dict[k].shape, state_dict[k].shape, msg))
        state_dict[k] = model_state_dict[k]
    else:
      print('Drop parameter {}.'.format(k) + msg)
  for k in model_state_dict:
    if not (k in state_dict):
      print('No param {}.'.format(k) + msg)
      state_dict[k] = model_state_dict[k]
  model.load_state_dict(state_dict, strict=False)

  # resume optimizer parameters
  if optimizer is not None and resume:
    if 'optimizer' in checkpoint:
      optimizer.load_state_dict(checkpoint['optimizer'])
      start_epoch = checkpoint['epoch']
      start_lr = lr
      for step in lr_step:
        if start_epoch >= step:
          start_lr *= 0.1
      for param_group in optimizer.param_groups:
        param_group['lr'] = start_lr
      print('Resumed optimizer with start lr', start_lr)
    else:
      print('No optimizer parameters in checkpoint.')
  if optimizer is not None:
    return model, optimizer, start_epoch
  else:
    return model
  

  # Resnet-18 expect normalized channels in input
class Normalize(object):
    def __init__(self):
        self.mean=[0.485, 0.456, 0.406]
        self.std=[0.229, 0.224, 0.225]
        self.norm = transforms.Normalize(self.mean, self.std)
    def __call__(self, image):
        image = image.astype(np.float32)/255
        axis = (0,1)
        image -= self.mean
        image /= self.std
        return image
    

from hardnet import get_pose_net
model = get_pose_net(85,{"hm":2,"offset":2,"wh":2,"angle":2})
#model = load_model(model,"centernet_hardnet85_coco.pth")

def resize_and_pad(image, target_size=(512, 512)):
    original_height, original_width = image.shape[:2]
    target_width, target_height = target_size

    # Calculate the scaling factor
    scale = min(target_width / original_width, target_height / original_height)
    
    # Calculate new dimensions
    new_width = int(original_width * scale)
    new_height = int(original_height * scale)
    
    # Resize the image
    resized_image = cv2.resize(image, (new_width, new_height))
    
    # Pad the image to the target size
    delta_w = target_width - new_width
    delta_h = target_height - new_height
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)
    padded_image = cv2.copyMakeBorder(resized_image, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0])
    
    return padded_image, scale, left, top


def pred4corner(hm,thresh=0.99):
    threshold = 0.2  # Adjust this threshold as needed
    _, thresholded_heatmap = cv2.threshold(hm, threshold, 1, cv2.THRESH_BINARY)
    
    # Find contours (connected components) in the thresholded heatmap
    contours, _ = cv2.findContours(thresholded_heatmap.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    keypoints = []
    for cnt in contours:
        # 2. Refine peak location (using contour center)
        try:
            M = cv2.moments(cnt)
            cx = int(M['m10'] / M['m00'])
            cy = int(M['m01'] / M['m00'])
            keypoints.append((cx, cy))
        except:
            continue
    return keypoints




fps = True
half = False
trace = True

model.load_state_dict(torch.load("4corner_hardnet_barcode_angle_great_results.pth"))
model.eval()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

if fps:
    from time import time

if half:
    model.half()
    
if trace:
    print("Start Tracing")
    rand_input = torch.rand(1,3,512,512).cuda()
    model = torch.jit.trace(model, rand_input)
    print("End Tracing")
    
cap = cv2.VideoCapture(0)
threshold = 0.2
while 1:
    ret,frame = cap.read()
    #image = cv2.resize(frame,(input_width,input_height))
    image = resize_and_pad(frame,target_size=(input_width,input_height))[0]
    #image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    img = image.copy()
    img = Normalize()(img)
    img = img.transpose([2,0,1])
    img = torch.from_numpy(img)
    if fps: f1 = time()
    if half:
        tensor = img.to(device).half().unsqueeze(0)
    else:
        tensor = img.to(device).float().unsqueeze(0)
    with torch.no_grad():
        hm, offset, wh, angle = model(tensor)
    if fps: f2 = time(); print("Fps:",1/(f2-f1))
    hm = hm.cpu().numpy().squeeze(0)#.squeeze(0)
    offset = offset.cpu().numpy().squeeze(0)
    wh = wh.cpu().numpy().squeeze(0)
    angle = angle.cpu().numpy().squeeze(0)
    hm = torch.sigmoid(torch.from_numpy(hm)).numpy()
    hm_corner = hm[1]
    hm_corner = cv2.resize(hm_corner,(image.shape[1],image.shape[0]))
    corners = pred4corner(hm_corner,0.2)
    for kp in corners:
        cv2.circle(image, kp, 5, (0, 0, 255), -1)
    hm = select(hm[0], threshold)
    sample = showbox(image, hm, offset, wh, angle, threshold)
    cv2.imshow("output",sample)
    ch = cv2.waitKey(1)
    if ch == ord("q"):
        cv2.destroyAllWindows()
        break
cap.release()


from sklearn.random_projection import SparseRandomProjection
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from sklearn.metrics import roc_auc_score
from torch.nn import functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import argparse
import torch
import cv2
import os
import time
import pickle

from tqdm import tqdm
from einops import rearrange, reduce, repeat
from PIL import Image
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from scipy.ndimage import gaussian_filter
from utils import *
from Mvtec import *
from model import *

def get_args():
    parser = argparse.ArgumentParser(description='ANOMALYDETECTION')
    parser.add_argument('--phase', choices=['train','test'], default='test')
    parser.add_argument('--dataset_path', default='./mvtec') # 'D:\Dataset\mvtec_anomaly_detection')#
    parser.add_argument('--result_path', default='./results/') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--category', default='metal_nut')
    parser.add_argument('--num_epochs', default=1)
    parser.add_argument('--batch_size', default=1)
    parser.add_argument('--load_size', default=256) 
    parser.add_argument('--input_size', default=224)
    parser.add_argument('--coreset_sampling_ratio', default=0.1)
    parser.add_argument('--project_root_path', default=r'./FAPM_memory/') # 'D:\Project_Train_Results\mvtec_anomaly_detection\210624\test') #
    parser.add_argument('--save_src_code', default=True)
    parser.add_argument('--save_anomaly_map', default=True)
    parser.add_argument('--n_neighbors', type=int, default=4)
    parser.add_argument('--save_img', type=bool, default=False)
    parser.add_argument('--score_threshold', type=float, default=0.5)
    parser.add_argument('--adaptive_ratio', type=float, default=2)
    args = parser.parse_args()
    return args

def calculate_patch_score(feats,near_m,far_m,selector,feat_num=4,h=7,w=7,n_neighbors=4):
    embedding_vec=rearrange(feats,'b c (h1 h) (w1 w)-> (h1 w1) (h w b) c', h1=h, w1=w)
    score=torch.zeros(49,feat_num,n_neighbors).cuda()
    score[selector]=torch.topk(torch.cdist(embedding_vec[selector],far_m[selector]),n_neighbors,dim=2,largest=False).values
    score[~selector]=torch.topk(torch.cdist(embedding_vec[~selector],near_m[~selector]),n_neighbors,dim=2,largest=False).values
    return score





def test(args,model,test_loader):
    gt_list_px_lvl=[]
    pred_list_px_lvl=[]
    gt_list_img_lvl=[]
    pred_list_img_lvl=[]
    img_path_list=[]
    inference_time_list=[]
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.eval()
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    embedding_dir_path, _, _ = prep_dirs(args.project_root_path)
    result_path=os.path.join(args.result_path,args.category)
    inv_normalize = transforms.Normalize(mean=[-0.485/0.229, -0.456/0.224, -0.406/0.255], std=[1/0.229, 1/0.224, 1/0.255])

    
    #### LOAD MEMORY BANK ####
    with open(embedding_dir_path+args.category+".pkl","rb") as f:
        embedding_dict=pickle.load(f)
    index_near_c= embedding_dict["final_near_core_c"]
    index_far_c= embedding_dict["final_far_core_c"]
    index_near_f = embedding_dict["final_near_core_f"]
    index_far_f = embedding_dict["final_far_core_f"]
    fn_selector_c = embedding_dict["fn_selector_c"]
    fn_selector_f = embedding_dict["fn_selector_f"]
    #### Memory on GPU ####
    if torch.cuda.is_available():
        index_near_f=torch.from_numpy(index_near_f).to(device)
        index_near_c=torch.from_numpy(index_near_c).to(device)
        index_far_f=torch.from_numpy(index_far_f).to(device)
        index_far_c=torch.from_numpy(index_far_c).to(device)
        fn_selector_c=torch.from_numpy(fn_selector_c).to(device).squeeze()
        fn_selector_f=torch.from_numpy(fn_selector_f).to(device).squeeze()
        print("memory on GPU")
    else:
        print("memory on CPU")
    model.eval()
    
    
    ###TEST###
    
    ##GPU WARM UP##
    dummy_input=torch.randn(1,3,224,224).cuda()
    print("GPU WARM UP")
    if args.save_img==True:
        os.makedirs(result_path,exist_ok=True)
    
    for (x, gt, label, file_name, x_type) in test_loader:
        x=x.to(device)
        gt=gt.to(device)
        label=label.to(device)
        _=model(x)
        score_patch_c=torch.zeros(49,4,args.n_neighbors).cuda() # Coarse Memory Score
        score_patch=torch.zeros(49,16,args.n_neighbors).cuda() # Fine Memory Score
    #####
    start=time.time()
    for (x, gt, label, file_name, x_type) in tqdm(test_loader):
        x=x.to(device)
        gt=gt.to(device)
        label=label.to(device)
        
        starter.record()  # start timing
        features = model(x)
        embeddings = []
        score_patch_c=torch.zeros(49,4,args.n_neighbors).cuda() # Coarse Memory Score
        score_patch=torch.zeros(49,16,args.n_neighbors).cuda() # Fine Memory Score
        for feature in features: 
            m = torch.nn.AvgPool2d(3, 1, 1)
            embeddings.append(m(feature)) 
        embedding_f = embeddings[0].cuda() # Fine Memory Bank
        embedding_c=embeddings[1].cuda() # Coarse Memory Bank
        score_patch_c=calculate_patch_score(embedding_c,index_near_c,index_far_c,fn_selector_c,4,7,7,args.n_neighbors) # Coarse Memory Score
        score_patch_c=repeat(score_patch_c,'b c n -> b (c 4) n')    # Coarse Memory Score
        score_patch_f=calculate_patch_score(embedding_f,index_near_f,index_far_f,fn_selector_f,16,7,7,args.n_neighbors) # Fine Memory Score
        score_patch=score_patch_c+score_patch_f #Total Memory Score
        anomaly_map = rearrange(score_patch[:,:,0], '(h1 h) (w1 w) -> (h1 w1) (h w)', h1=7, w1=4).cpu().detach().numpy()
        score_patches = rearrange(score_patch, '(h1 h) (w1 w) v-> (h1 w1 h w) v', h1=7,w1=4).cpu().detach().numpy()
        N_b = score_patches[np.argmax(score_patches[:,0])]#/mean_dis[np.argmax(score_patches[:,0])]
        w = (1 - (np.max(np.exp(N_b))/np.sum(np.exp(N_b))))
        score = w*max(score_patches[:,0]) # Image-level score
        
        
        gt_np = gt.cpu().numpy()[0,0].astype(int)
        anomaly_map_resized = cv2.resize(anomaly_map, (args.input_size, args.input_size))
        anomaly_map_resized_blur = gaussian_filter(anomaly_map_resized, sigma=4)
        
        ender.record() # end timing
        torch.cuda.synchronize() # wait for cuda to finish (cuda is asynchronous!)
        inference_time = starter.elapsed_time(ender)*1e-3
        
        gt_list_px_lvl.extend(gt_np.ravel())
        pred_list_px_lvl.extend(anomaly_map_resized_blur.ravel())
        gt_list_img_lvl.append(label.cpu().numpy()[0])
        pred_list_img_lvl.append(score)
        img_path_list.extend(file_name)
        inference_time_list.append(inference_time)
        
        if args.save_img==True:
            x = inv_normalize(x)
            input_x = cv2.cvtColor(x.permute(0,2,3,1).cpu().numpy()[0]*255, cv2.COLOR_BGR2RGB)
            save_anomaly_map(result_path,anomaly_map_resized_blur, input_x, gt_np*255, file_name[0], x_type[0])
    
    
    end=time.time()
    print("category:",args.category)
    fps=len(img_path_list)/(end-start)
    print("fps:",fps)
    avg_time=sum(inference_time_list)/len(img_path_list)
    print("average_inerence_time:",avg_time)
    pixel_auc = roc_auc_score(gt_list_px_lvl, pred_list_px_lvl)
    print("Total pixel-level auc-roc score :",pixel_auc)
    img_auc = roc_auc_score(gt_list_img_lvl, pred_list_img_lvl)
    print("Total image-level auc-roc score :",img_auc)
    
    print('test_epoch_end')
    values = {'img_auc': img_auc, 'pixel_auc': pixel_auc, "Fps": fps}
    return fps,pixel_auc,img_auc
        
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    mean_train = [0.485, 0.456, 0.406]
    std_train = [0.229, 0.224, 0.225]
    args=get_args()
    data_transforms=transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size), Image.ANTIALIAS),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size),
                        transforms.Normalize(mean=mean_train,
                                            std=std_train)])
    gt_transforms=transforms.Compose([
                        transforms.Resize((args.load_size, args.load_size)),
                        transforms.ToTensor(),
                        transforms.CenterCrop(args.input_size)])
    test_datasets = MVTecDataset(root=os.path.join(args.dataset_path,args.category), transform=data_transforms, gt_transform=gt_transforms, phase='test')
    test_loader = DataLoader(test_datasets, batch_size=1, shuffle=False, num_workers=20) #, pin_memory=True) # only work on batch_size=1, now.
    model = STPM().cuda()
    fps,pixel_auc,img_auc=test(args,model,test_loader)


if __name__=='__main__':
    main()

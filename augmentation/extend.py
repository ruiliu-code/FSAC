import torch
import scipy.misc
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
def extract_ampl_phase(fft_im):
    fft_amp = fft_im[:,:,:,0]**2 + fft_im[:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,1], fft_im[:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate_add( amp_src, amp_trg1, amp_trg2, L=0.1 ):
    _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*L)).astype(int) 
    weight = np.random.rand(4,1)
    if b < 1:
        b += 1
    try:
        # and
        amp_src[:,0:b,0:b]     = weight[0,0] * amp_trg1[:,0:b,0:b] + ( 1 - weight[0,0] ) * amp_trg2[:,0:b,0:b] # top left
        amp_src[:,0:b,w-b:w]   = weight[1,0] * amp_trg1[:,0:b,w-b:w] + ( 1 - weight[1,0] ) * amp_trg2[:,0:b,w-b:w] # top right
        amp_src[:,h-b:h,0:b]   = weight[2,0] * amp_trg1[:,h-b:h,0:b] + ( 1 - weight[2,0] ) * amp_trg2[:,h-b:h,0:b] # bottom left    
        amp_src[:,h-b:h,w-b:w] = weight[3,0] * amp_trg1[:,h-b:h,w-b:w] + ( 1 - weight[3,0] ) * amp_trg2[:,h-b:h,w-b:w] # bottom right
    except:
        import ipdb;ipdb.set_trace()

    return amp_src

def FDA(src_img, trg_img1, trg_img2, L=0.1):
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg1 = torch.rfft( trg_img1.clone(), signal_ndim=2, onesided=False )
    fft_trg2 = torch.rfft( trg_img2.clone(), signal_ndim=2, onesided=False )
    #print(fft_src.shape)
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg1, pha_trg1 = extract_ampl_phase( fft_trg1.clone())
    amp_trg2, pha_trg2 = extract_ampl_phase( fft_trg2.clone())

    amp_src_ = low_freq_mutate_add( amp_src.clone(), amp_trg1.clone(), amp_trg2.clone(), L=L )

    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    _,imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg

mode = 'extend'
src_dir = 'your data path/VOC2007/JPEGImages/'
trg_dir = 'your data path/VOC2007/JPEGImages/'

save_dir = 'save path'

src_list = os.listdir(src_dir)
trg_list = os.listdir(trg_dir)

np.random.seed(217)

for i in tqdm(range(len(src_list))):
    src_name = src_list[i]
    im_src = Image.open(src_dir + src_name).convert('RGB')
    im_src = torch.tensor(np.asarray(im_src, np.float32)).cuda()
    src_shape = im_src.shape
    im_src = im_src.permute(2, 0, 1)
    
    idx = np.random.randint(0,len(trg_list),(2,1))
    #print('idx is {}'.format(idx))
    trg_name1 = trg_list[idx[0,0]]
    trg_name2 = trg_list[idx[1,0]]


    im_trg1 = Image.open(trg_dir + trg_name1).convert('RGB')
    im_trg2 = Image.open(trg_dir + trg_name2).convert('RGB')


    im_trg1 = im_trg1.resize( (src_shape[1], src_shape[0]), Image.BICUBIC) #!!!!!!!!!!!!!!!!!!
    im_trg2 = im_trg2.resize( (src_shape[1], src_shape[0]), Image.BICUBIC) #!!!!!!!!!!!!!!!!!!

    im_trg1 = torch.tensor(np.asarray(im_trg1, np.float32)).cuda()
    im_trg2 = torch.tensor(np.asarray(im_trg2, np.float32)).cuda()

    try:
        assert im_trg1.shape==src_shape
        assert im_trg2.shape==src_shape
    except:
        import ipdb;ipdb.set_trace()

    im_trg1 = im_trg1.permute(2, 0, 1)
    im_trg2 = im_trg2.permute(2, 0, 1)

    src_in_trg = FDA(im_src, im_trg1, im_trg2, L = 0.01)
    assert src_in_trg.shape == im_src.shape
    scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(save_dir + src_name)



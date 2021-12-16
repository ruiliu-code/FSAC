import torch
import scipy.misc
from PIL import Image
import numpy as np
import os
from tqdm import tqdm
import random
def extract_ampl_phase(fft_im):
    fft_amp = fft_im[:,:,:,0]**2 + fft_im[:,:,:,1]**2
    fft_amp = torch.sqrt(fft_amp)
    fft_pha = torch.atan2( fft_im[:,:,:,1], fft_im[:,:,:,0] )
    return fft_amp, fft_pha

def low_freq_mutate_c( amp_src, amp_trg, L=0.1 ):
    _, h, w = amp_src.size()
    c1 = [0,1,2]
    c2 = [1,0,2]
    random.shuffle(c1)
    random.shuffle(c2)
    b = (np.floor(np.amin((h,w))*L)).astype(int) 
    if b < 1:
        b += 1
    try:
        # in
        amp_src[c1[0],0:b,0:b]     = amp_trg[c2[0],0:b,0:b] # top left
        amp_src[c1[0],0:b,w-b:w]   = amp_trg[c2[0],0:b,w-b:w] # top right
        amp_src[c1[0],h-b:h,0:b]   = amp_trg[c2[0],h-b:h,0:b] # bottom left    
        amp_src[c1[0],h-b:h,w-b:w] = amp_trg[c2[0],h-b:h,w-b:w]  # bottom right

        amp_src[c1[1],0:b,0:b]     = amp_trg[c2[1],0:b,0:b] # top left
        amp_src[c1[1],0:b,w-b:w]   = amp_trg[c2[1],0:b,w-b:w] # top right
        amp_src[c1[1],h-b:h,0:b]   = amp_trg[c2[1],h-b:h,0:b] # bottom left    
        amp_src[c1[1],h-b:h,w-b:w] = amp_trg[c2[1],h-b:h,w-b:w]  # bottom right

        amp_src[c1[2],0:b,0:b]     = amp_trg[c2[2],0:b,0:b] # top left
        amp_src[c1[2],0:b,w-b:w]   = amp_trg[c2[2],0:b,w-b:w] # top right
        amp_src[c1[2],h-b:h,0:b]   = amp_trg[c2[2],h-b:h,0:b] # bottom left    
        amp_src[c1[2],h-b:h,w-b:w] = amp_trg[c2[2],h-b:h,w-b:w]  # bottom right
    except:
        import ipdb;ipdb.set_trace()

    return amp_src

def low_freq_mutate_in( amp_src, amp_trg, L=0.1 ):
    _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*L)).astype(int) 
    if b < 1:
        b += 1
    try:
        # in
        amp_src[:,0:b,0:b]     = amp_trg[:,0:b,0:b] # top left
        amp_src[:,0:b,w-b:w]   = amp_trg[:,0:b,w-b:w] # top right
        amp_src[:,h-b:h,0:b]   = amp_trg[:,h-b:h,0:b] # bottom left    
        amp_src[:,h-b:h,w-b:w] = amp_trg[:,h-b:h,w-b:w]  # bottom right
    except:
        import ipdb;ipdb.set_trace()

    return amp_src

def low_freq_mutate_and( amp_src, amp_trg, L=0.1 ):
    _, h, w = amp_src.size()
    b = (np.floor(np.amin((h,w))*L)).astype(int) 
    weight = np.random.rand(4,1)
    if b < 1:
        b += 1
    try:
        # and
        amp_src[:,0:b,0:b]     = weight[0,0] * amp_src[:,0:b,0:b] + (1 - weight[0,0]) * amp_trg[:,0:b,0:b]     # top left
        amp_src[:,0:b,w-b:w]   = weight[1,0] * amp_src[:,0:b,w-b:w] + (1 - weight[1,0]) * amp_trg[:,0:b,w-b:w] # top right
        amp_src[:,h-b:h,0:b]   = weight[2,0] * amp_src[:,h-b:h,0:b] + (1 - weight[2,0]) * amp_trg[:,h-b:h,0:b] # bottom left    
        amp_src[:,h-b:h,w-b:w] = weight[3,0] * amp_src[:,h-b:h,w-b:w] + (1 - weight[3,0]) * amp_trg[:,h-b:h,w-b:w]  # bottom right
    except:
        import ipdb;ipdb.set_trace()

    return amp_src


def FDA(src_img, trg_img, L=0.1, mode='in'):
    fft_src = torch.rfft( src_img.clone(), signal_ndim=2, onesided=False ) 
    fft_trg = torch.rfft( trg_img.clone(), signal_ndim=2, onesided=False )
    #print(fft_src.shape)
    amp_src, pha_src = extract_ampl_phase( fft_src.clone())
    amp_trg, pha_trg = extract_ampl_phase( fft_trg.clone())

    if mode == 'replace':
        amp_src_ = low_freq_mutate_in( amp_src.clone(), amp_trg.clone(), L=L )
    elif mode == 'mix':
        amp_src_ = low_freq_mutate_and( amp_src.clone(), amp_trg.clone(), L=L )
    elif mode == 'disorder':
        amp_src_ = low_freq_mutate_c( amp_src.clone(), amp_trg.clone(), L=L )

    fft_src_ = torch.zeros( fft_src.size(), dtype=torch.float )
    fft_src_[:,:,:,0] = torch.cos(pha_src.clone()) * amp_src_.clone()
    fft_src_[:,:,:,1] = torch.sin(pha_src.clone()) * amp_src_.clone()

    _,imgH, imgW = src_img.size()
    src_in_trg = torch.irfft( fft_src_, signal_ndim=2, onesided=False, signal_sizes=[imgH,imgW] )

    return src_in_trg



trg_dir = 'your data path/VOC2007/JPEGImages/'
src_dir = 'your data path/VOC2007/JPEGImages/'
mode = 'disorder' # 'mix' or 'replace'

save_dir = 'save path'

src_list = os.listdir(src_dir)
trg_list = os.listdir(trg_dir)


np.random.seed(217)

for i in tqdm(range(len(src_list))):
    # src_name = src_list[i]
    src_name = src_list[i][:-4] + '.jpg'
    im_src = Image.open(src_dir + src_name).convert('RGB')
    im_src = torch.tensor(np.asarray(im_src, np.float32)).cuda()
    src_shape = im_src.shape
    im_src = im_src.permute(2, 0, 1)
    
    idx = np.random.randint(0,len(trg_list))
    #print('idx is {}'.format(idx))
    trg_name = trg_list[idx]
    im_trg = Image.open(trg_dir + trg_name).convert('RGB')
    im_trg = im_trg.resize( (src_shape[1], src_shape[0]), Image.BICUBIC) #!!!!!!!!!!!!!!!!!!
    im_trg = torch.tensor(np.asarray(im_trg, np.float32)).cuda()
    try:
        assert im_trg.shape==src_shape
    except:
        import ipdb;ipdb.set_trace()
    im_trg = im_trg.permute(2, 0, 1)

    src_in_trg = FDA(im_src, im_trg, L = 0.01, mode = mode)
    assert src_in_trg.shape == im_src.shape
    scipy.misc.toimage(src_in_trg, cmin=0.0, cmax=255.0).save(save_dir + src_name)
    if i > 64:
        import ipdb;ipdb.set_trace()

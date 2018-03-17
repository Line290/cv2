
import os
import time
import subprocess
import cv2
import numpy as np
import matplotlib.pylab as plt
from scipy import signal

def img2lab(img_path):
    bgr_img = cv2.imread(img_path)
    b,g,r = cv2.split(bgr_img)
    rgb_img = cv2.merge([r,g,b])
    return cv2.cvtColor(rgb_img, cv2.COLOR_BGR2LAB)

def pad_img(img, kernel_size, padding):
    '''input: img: a NumPy array, get from function img2lab
              kernel_size: an int number, maybe 3*3 or 5*5 or 7*7 etc. 7*7 in paper
              padding: the ways of padding
       output:img: a image after padding
    '''
    size = (kernel_size - 1) / 2
    if padding == 'replicate':
        img_pad = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_REPLICATE)
    elif padding == 'constant':
        WHITE = [0, 0, 0]
        img_pad = cv2.copyMakeBorder(img, size, size, size, size, cv2.BORDER_CONSTANT, value=WHITE)
    return img_pad

def pixel_patch_vectorize(img_pad, kernel_size, stride):
    H, W, C = img_pad.shape
    s0, s1, s2 = img_pad.strides
    nH = H - kernel_size + 1
    nW = W - kernel_size + 1
    # In there kernel size is a square in all 3 channels
    # Actually, the size of kernel is  kernel_size*kernel_size*3
    nC = C - 3 + 1
    
    shp = kernel_size, kernel_size, C, nH, nW, nC
    strd = s0, s1, s2, s0, s1, s2
    out_view = np.lib.stride_tricks.as_strided(img_pad, shape=shp, strides=strd)
    img_patch_full = out_view.reshape(kernel_size*kernel_size*C, nH, nW).transpose(1,2,0)
    H_idx = np.array(range(0, nH, stride)).reshape(-1,1)
    W_idx = np.array(range(0, nW, stride)).reshape(1,-1)
    return img_patch_full[H_idx, W_idx]

def get_saliency_value_vectorize(img_patch, C, K):
    # C: the weight multiply by positional distance
    # K: the top K similar distances with other pixel patches for each pixel
    H, W, C = img_patch.shape
    # save the saliency value
    s = np.zeros((H, W))

    # get matrix's shape A:(H,W,H,W,kernel_size*kernel_size*C)
    # A[i,j,:,:,:] is img_patch
    A = np.tile(img_patch, (H,W,1)).reshape(H,H,W,W,C).transpose(0,2,1,3,4)

    # get matrix's shape B:(H,W,H,W,kernel_size*kernel_size*C)
    # B[i,j,:,:,:] is img_patch[i,j,:]
    B = np.repeat(img_patch, H, axis=0)
    B = np.repeat(B, W, axis=1).reshape(H,H,W,W,C).transpose(0,2,1,3,4)

    # calculate the color distances for all pixels with other
    d_col = np.sqrt(np.sum((A - B)**2, axis=4))
    # normlized to [0,1]
    d_col_norm = (d_col - d_col.min((2,3))) / (d_col.max((2,3)) - d_col.min((2,3)))
    
    # find K most similar patches according to d_col value
    
      # add a very small perturbation, avoid too many value equal to 0
    d_col_norm_flat = d_col_norm.reshape(H*W, -1) + 0.0000001*np.random.rand(H*W, H*W)
    # get the Kth value for each pixel
    Kth = d_col_norm_flat[range(H*W), np.argpartition(d_col_norm_flat,K)[:, K:K+1].reshape(-1)].reshape(-1,1)
      # shape (H*W*K, 2)   
    Kth_idx = np.argwhere(d_col_norm_flat < Kth)

      # shape (H*W, K)
    d_col_K = d_col_norm_flat[Kth_idx[:,0], Kth_idx[:,1]].reshape(H*W, -1)
    
    # calculate positional distance for all pixels of the top K
      # shape (H*W, K)
    Kth_x_idx = Kth_idx[:,1].reshape(H*W, -1) / W
    Kth_y_idx = Kth_idx[:,1].reshape(H*W, -1) % W
      # pixel idx, shape(H*W, 1)
    pixel_x_idx = np.repeat(np.array(range(H)), W)
    pixel_y_idx = np.tile(np.array(range(W)), H)

    d_pos = np.sqrt((Kth_x_idx - pixel_x_idx.reshape(-1,1))**2 + (Kth_y_idx - pixel_y_idx.reshape(-1,1))**2)
    # normalized by the larger image dimension
    d_pos_norm = d_pos / max(H,W)
    
    # single-scale saliency value map of a image at scale r
    dis_K = d_col_K / (1 + C*d_pos_norm)
    s_flat = 1 -  np.exp(-1.0/K * np.sum(dis_K, axis=1))
    s = s_flat.reshape(H, W)
    
    # # normalized to [0, 1]
    s_norm = (s - s.min()) / (s.max() - s.min())
    return s_norm

def contextual_effect_vectorize(sal_map, threshold):
    # threshold: Threshold is 0.8 in paper.
    H, W = sal_map.shape
    s_attended_idx = np.argwhere(sal_map>threshold)
    
      # pixel idx, shape(H*W, 2)
    pixel_x_idx = np.repeat(np.array(range(H)), W)
    pixel_y_idx = np.tile(np.array(range(W)), H)
    
    d_foci = np.sqrt((pixel_x_idx.reshape(-1,1) - s_attended_idx[:,0].reshape(1,-1))**2+
                            (pixel_y_idx.reshape(-1,1) - s_attended_idx[:,1].reshape(1,-1))**2)
    d_foci = d_foci.min(axis=1).reshape(H, W)
    
    # normalized to the range [0,1]
    d_foci_norm = (d_foci - d_foci.min()) / (d_foci.max() - d_foci.min())
    # Apply to saliency map
    sal_cont_map = sal_map * (1 - d_foci_norm)
    return sal_cont_map

def multiscale_vectorize(img_path, img_scales, kernel_size, stride, C, K, threshold, padding):
    # img_scales: a list, the rates of shrink a image. In paper, it is [1, 0.8, 0.5, 0.3]
    # get image in LAB type
    img = img2lab(img_path)
    H_org, W_org, _ = img.shape
    rate = H_org*1.0 / W_org

    # resize the resolution of image, the maximum is 250    
    if H_org > W_org and H_org > 250:
        H = 250
        W = int(H / rate)
    elif W_org > H_org and W_org > 250:
        W = 250
        H = int(W*rate)
    else:
        W, H = W_org, H_org
    
    img = cv2.resize(img, (W, H))
    H, W, _ = img.shape
    # multi-scale saliency value
    N = len(img_scales)
    s = np.zeros((H, W, N))
    
    for i, scale_rate in enumerate(img_scales):
        # shrink image at a scale_rate
        img_scale = cv2.resize(img, (int(W*scale_rate), int(H*scale_rate)))
        
        # for each kernel size, do padding
        img_pad = pad_img(img_scale, kernel_size, padding)
        
        # get patch matrix for each pixel in image
        img_patch_verc = pixel_patch_vectorize(img_pad, kernel_size, stride)

        # get saliency value for a image scale
        sal_map = get_saliency_value_vectorize(img_patch_verc, C, K)
    
        # Interpolated back to original image size, default value is bilinear interpolation.
        sal_map_rescale = cv2.resize(sal_map, (W, H))
        
        # Immediate Context
#         sal_cont_map = contextual_effect(sal_map_rescale, threshold)
        sal_cont_map = contextual_effect_vectorize(sal_map_rescale, threshold)
    
        s[:,:,i] = sal_cont_map
    s_aver = 1.0/N * np.sum(s, axis = 2)
    return s_aver

def center_prior(s):
    from scipy import signal
    H, W = s.shape
    H_std = H*1.0/6
    W_std = W*1.0/6
    gH = signal.gaussian(H, std = H_std).reshape(H, 1)
    gW = signal.gaussian(W, std = W_std).reshape(W, 1)
    g2d = np.outer(gH, gW)
    return g2d*s
#     print g2d.shape
#     plt.imshow(g2d, cmap='gray')
#     plt.axis("off")
#     plt.show()

def main():
    PATH = './hw2_test_images'
    print 'Check image files'
    print subprocess.check_output(["ls", PATH]).decode('utf8')
    all_imgs = os.listdir(PATH)
    for img in all_imgs:
        if img == '.ipynb_checkpoints':
            continue
        img_path = os.path.join(PATH, img)
        print img_path

        kernel_size = 7
        img_scales = [1, 0.8, 0.5, 0.3]
        # img_scales = [1]
        threshold = 0.8
        C = 3
        K = 65
        stride = 3
        padding = 'replicate'
        start  = time.time()
        saliency_map = multiscale_vectorize(img_path, 
                                            img_scales, 
                                            kernel_size, 
                                            stride, 
                                            C, 
                                            K, 
                                            threshold, 
                                            padding)
        end = time.time()
        print 'Running time is: '+str(end-start)
        s = center_prior(saliency_map)
        
        # show original image
        bgr_img = cv2.imread(img_path)
        rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
        plt.imshow(rgb_img)
        plt.axis("off")
        plt.show()
        # show saliency map
        plt.imshow(saliency_map, cmap='gray')
        plt.axis("off")
        plt.show()
        # show saliency map with center prior
        plt.imshow(s, cmap='gray')
        plt.axis("off")
        plt.show()
#         cv2.imwrite(PATH+'/saliency'+img, saliency_map)

if __name__ == '__main__':
    main()

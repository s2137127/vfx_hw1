import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
from argparse import ArgumentParser
from HDR_alg import *

parse = ArgumentParser('High Dynamic Range Imaging')
parse.add_argument('--tone_map',default='global',type=str,choices=['global','local'],help='tone mapping algorithm')
parse.add_argument('--save_img',default=True,type=bool,help='save tone mapping image')
parse.add_argument('--save_dir',default='./rgb_img.jpg',type=str,help='dir to save img')
parse.add_argument('--plot_response',default=False,type=bool,help='plot response curve')
parse.add_argument('--show_radiance',default=True,type=bool,help='show radiance map')
parse.add_argument('--HDR_alg',default='Debvec',type=str,choices=['Robertson','Debvec'],help='HDR alogorithm')
zmin=0
zmax=255
img_name_list = ['./img1/img%02d.jpg' %i for i in range(1,14) ]
# img_name_list = ['./img2/memorial%04d.png' %i for i in range(61,77) ]
P = 256 #p個點
N = len(img_name_list) #N張影像
n = 256
global g
Zij = np.zeros((P,N,3),np.int16)#[point,pic,channel]

B = np.array([np.log(13),np.log(10),np.log(4),np.log(3.2),np.log(1),np.log(0.8),np.log(0.3),np.log(0.25),np.log(1/60)
              ,np.log(1/80),np.log(1/320),np.log(1/400),np.log(1/1000)],dtype=np.float64)#log delta t
# B = np.array([np.log(1/0.03125),np.log(1/0.0625),np.log(1/0.125),np.log(1/0.25),np.log(1/0.5),np.log(1),np.log(1/2),np.log(1/4),np.log(1/8)
#               ,np.log(1/16),np.log(1/32),np.log(1/64),np.log(1/128),np.log(1/256),np.log(1/512),np.log(1/1024)])

Weight = np.float32([z-zmin+1 if z<(zmin+zmax)/2 else zmax-z+1 for z in range(256) ])


def lnE(Zij,n):
    sumup = 0.0
    sumdown = 0.0
    for i in range(Zij.shape[0]):  
        w = Weight[Zij[i]]   
        sumup += w*(g[Zij[i],n]-B[i])
        sumdown += w
    
    
    lnEi = sumup/sumdown
    
    return lnEi
def tone_mapping_global(hdr,a=0.25,L_white=180):
    Ld = np.zeros((hdr.shape[0],hdr.shape[1]),dtype=np.float32)
    out = np.zeros(hdr.shape,dtype=np.float32)
    delta = np.ones((hdr.shape[0],hdr.shape[1]),dtype=np.float32)*1e-6
   
    Lw = 0.2*hdr[:,:,2] + 0.7*hdr[:,:,1]+0.1*hdr[:,:,0]
    one = np.ones_like(Lw)
    Lw_avg = np.exp(np.sum(np.log(Lw+delta))/(hdr.shape[0]*hdr.shape[1]))
    Lm = (a/Lw_avg)*Lw
    Ld = Lm*(one+(Lm/L_white**2))/(one+Lm)
               
    out[:,:,0] = Ld*hdr[:,:,0]/Lw
    out[:,:,1] = Ld*hdr[:,:,1]/Lw
    out[:,:,2] = Ld*hdr[:,:,2]/Lw
    out *= 255
    return out.clip(0,255).astype(np.uint8)
def gaussian_blurs(im, smax=25, a=0.25, fi=8.0, epsilon=0.01):
    cols, rows = im.shape
    blur_prev = im
    num_s = int((smax+1)/2)
    
    blur_list = np.zeros(im.shape + (num_s,))
    Vs_list = np.zeros(im.shape + (num_s,))
    
    for i, s in enumerate(range(1, smax+1, 2)):
        blur = cv2.GaussianBlur(im, (s, s), 0)
        Vs = np.abs((blur - blur_prev) / (2 ** fi * a / s ** 2 + blur_prev))
        blur_list[:, :, i] = blur
        Vs_list[:, :, i] = Vs

    smax = np.argmax(Vs_list > epsilon, axis=2)
    smax[np.where(smax == 0)] = num_s
    smax -= 1

    I, J = np.ogrid[:cols, :rows]
    blur_smax = blur_list[I, J, smax]
    
    return blur_smax
def tone_mapping_local(hdr,a=0.25):
   
    Ld = np.zeros((hdr.shape[0],hdr.shape[1]),dtype=np.float32)
    out = np.zeros(hdr.shape,dtype=np.float32)
    delta = np.ones((hdr.shape[0],hdr.shape[1]),dtype=np.float32)*1e-6
    Lw = 0.2*hdr[:,:,2] + 0.7*hdr[:,:,1]+0.1*hdr[:,:,0]
    # print(Lw)
    Lw_avg = np.exp(np.sum(np.log(Lw+delta))/(hdr.shape[0]*hdr.shape[1]))
    # print(Lw_avg)
    Lm = (a/Lw_avg)*Lw
    L_smax = gaussian_blurs(Lm)
    Ld = Lm/(1+L_smax)
           
    out[:,:,0] = Ld*hdr[:,:,0]/Lw
    out[:,:,1] = Ld*hdr[:,:,1]/Lw
    out[:,:,2] = Ld*hdr[:,:,2]/Lw
    out *= 255
    return out.clip(0,255).astype(np.uint8)
def draw_radiance_map(img_list):
    img_tmp =np.array(img_list)
    # print(img_tmp.shape)
    out = np.zeros((img_tmp.shape[1],img_tmp.shape[2],3),dtype=np.float32)
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j,0] = lnE(img_tmp[:,i,j,0],0)
            out[i,j,1] = lnE(img_tmp[:,i,j,1],1)
            out[i,j,2] = lnE(img_tmp[:,i,j,2],2)
   
    cv2.imwrite("./test.hdr",np.exp(out).astype(np.float32))
    return np.exp(out)
   
if __name__ == '__main__':
    
    args = vars(parse.parse_args())

    img_list = [cv2.imread(i) for i in img_name_list]
    rand_pick = np.random.randint(low=0,high=[img_list[0].shape[0],img_list[0].shape[1]],size = (P,2))
    
    for i in range(len(img_list)):
        for j in range(P):
            Zij[j,i,0] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],0])
            Zij[j,i,1] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],1])
            Zij[j,i,2] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],2])
    
    l=0
    if args['HDR_alg'] == 'Robertson':
        R = Robertson(Zij,B,Weight)
        g = R.solve()
    elif args['HDR_alg'] == 'Debvec' :
        D = Debvec(Zij,B,Weight,N)
        g = D.solve()
    hdr = draw_radiance_map(img_list)
    if args['show_radiance']:
        fig = plt.figure()
        plt.subplot(131)
        plt.imshow(np.log(hdr[...,0]).astype(int), cmap="jet", origin="upper")
        plt.subplot(132)
        plt.imshow(np.log(hdr[...,1]).astype(int), cmap="jet", origin="upper")
        plt.subplot(133)
        plt.imshow(np.log(hdr[...,2]).astype(int), cmap="jet", origin="upper")
        # plt.colorbar()
        plt.show()

    if args['tone_map'] == 'global':
        out = tone_mapping_global(hdr,a=0.25,L_white=255)
    elif args['tone_map'] == 'local':
        out = tone_mapping_local(hdr)
    if args['save_img']:
        cv2.imwrite(args['save_dir'],out)
    # lnE = np.array([x[i,:] for i in range(256,x.shape[0])])
    if args['plot_response']:
        plt.plot(g[:,0],range(0,256),color='blue') 
        plt.plot(g[:,1],range(0,256),color='green') 
        plt.plot(g[:,2],range(0,256),color='red') 
        plt.show()
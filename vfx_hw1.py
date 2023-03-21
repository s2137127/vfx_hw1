import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
zmin=0
zmax=255
img_name_list = ['./img1/img%02d.jpg' %i for i in range(1,14) ]
# img_name_list = ['./img2/memorial%04d.png' %i for i in range(61,77) ]
P = 256 #p個點
N = len(img_name_list) #N張影像
n = 256
Zij = np.zeros((P,N,3),np.int16)#[point,pic,channel]
A = np.zeros((N*P+254+1,n+P,3),dtype=np.float64)
b = np.zeros((A.shape[0],1,3),dtype=np.float64)
x = np.zeros((256+P,3),dtype=np.float64)
Wij = 0
B = np.array([np.log(13),np.log(10),np.log(4),np.log(3.2),np.log(1),np.log(0.8),np.log(0.3),np.log(0.25),np.log(1/60)
              ,np.log(1/80),np.log(1/320),np.log(1/400),np.log(1/1000)],dtype=np.float64)#log delta t
# B = np.array([np.log(1/0.03125),np.log(1/0.0625),np.log(1/0.125),np.log(1/0.25),np.log(1/0.5),np.log(1),np.log(1/2),np.log(1/4),np.log(1/8)
#               ,np.log(1/16),np.log(1/32),np.log(1/64),np.log(1/128),np.log(1/256),np.log(1/512),np.log(1/1024)])

Weight = np.float32([z-zmin+1 if z<(zmin+zmax)/2 else zmax-z+1 for z in range(256) ])

def solve_svd(A,b):
    # compute svd of A
    # U,s,Vh = np.linalg.svd(A,full_matrices=False)

    # # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    # c = np.dot(U.T,b)
    # # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    # w = np.dot(np.diag(1/s),c)
    # # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    # x = np.dot(Vh.conj().T,w)
    inv_A = np.linalg.pinv(A)
    x = np.dot(inv_A, b)
    return x.astype(np.float32)
def lnE(Zij,n):
    sumup = 0.0
    sumdown = 0.0
    for i in range(Zij.shape[0]):  
        w = Weight[Zij[i]]   
        sumup += w*(g[Zij[i],n]-B[i])
        sumdown += w
    
    
    lnEi = sumup/sumdown
    
    return lnEi
def tone_mapping(hdr,a=3,L_white=180):
    Ld = np.zeros((hdr.shape[0],hdr.shape[1]),dtype=np.float32)
    out = np.zeros(hdr.shape,dtype=np.float32)
    delta = np.ones((hdr.shape[0],hdr.shape[1]),dtype=np.float32)*1e-6
    Lw = 0.2*hdr[:,:,2] + 0.7*hdr[:,:,1]+0.1*hdr[:,:,0]
    # print(Lw)
    Lw_avg = np.exp(np.sum(np.log(Lw+delta))/(hdr.shape[0]*hdr.shape[1]))
    print(Lw_avg)
    Lm = (a/Lw_avg)*Lw
    for i in range(Ld.shape[0]):
        for j in range(Ld.shape[1]):
            Ld[i,j] = Lm[i,j]*(1+(Lm[i,j]/L_white**2))/(1+Lm[i,j])
            # Ld[i,j] = Lm[i,j]*((1+Lm[i,j])/(1+Lm[i,j]))
           
    out[:,:,0] = Ld*hdr[:,:,0]/Lw
    out[:,:,1] = Ld*hdr[:,:,1]/Lw
    out[:,:,2] = Ld*hdr[:,:,2]/Lw
    print(out)
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
    # plt.imshow(out[...,2].astype(int), cmap="jet", origin="upper")
    # plt.colorbar()
    # plt.show()
if __name__ == '__main__':
   
    img_list = [cv2.imread(i) for i in img_name_list]
    rand_pick = np.random.randint(low=0,high=[img_list[0].shape[0],img_list[0].shape[1]],size = (P,2))
 
    for i in range(len(img_list)):
        for j in range(P):
            Zij[j,i,0] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],0])
            Zij[j,i,1] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],1])
            Zij[j,i,2] = int(img_list[i][rand_pick[j,0],rand_pick[j,1],2])
    
    l=0
    for i in range(Zij.shape[0]):
        for j in range(Zij.shape[1]):
            for k in range(Zij.shape[2]):
                Wij = Weight[Zij[i,j,k]]
                #print(Wij)
                A[l,Zij[i,j,k],k] = Wij
                A[l,n+i,k] = -Wij
                b[l,0,k] = Wij*B[j]
            l += 1

    A[l,127,:] = 1
    l += 1
    for i in range(0,254):
        A[l,i,:] = Weight[i+1]
        A[l,i+1,:] = -2*Weight[i+1]
        A[l,i+2,:] = Weight[i+1]
        l += 1
    x[:,0] = solve_svd(A[:,:,0],b[:,:,0]).squeeze()
    x[:,1] = solve_svd(A[:,:,1],b[:,:,1]).squeeze()
    x[:,2] = solve_svd(A[:,:,2],b[:,:,2]).squeeze()
    g = np.array([x[i,:] for i in range(256)])
    hdr = draw_radiance_map(img_list)
    out = tone_mapping(hdr,a=0.25,L_white=180)
    cv2.imshow("rbg",out)
    cv2.waitKey(0)
    # lnE = np.array([x[i,:] for i in range(256,x.shape[0])])

    plt.plot(g[:,0],range(0,256)) 
    plt.show()
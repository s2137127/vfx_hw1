import numpy as np
import cv2
import matplotlib.pyplot as plt
zmin=0
zmax=255
#img_name_list = ['img01.jpg','img02.jpg','img03.jpg']
img_name_list = ['img%02d.jpg' %i for i in range(1,14) ]
P = 100 #p個點
N = len(img_name_list) #N張影像
n = 256
Zij = np.zeros((P,N,3),np.uint8)
A = np.zeros((N*P+254+1,n+P,3))
b = np.zeros((A.shape[0],1,3))
x = np.zeros((256+P,3))
Wij = 0
B = np.array([np.log(13),np.log(10),np.log(4),np.log(3.2),np.log(1),np.log(0.8),np.log(0.3),np.log(0.25),np.log(1/60)
              ,np.log(1/80),np.log(1/320),np.log(1/400),np.log(1/1000)])#log delta t
Weight = [z-zmin+1 if z<(zmin+zmax)/2 else zmax-z+1 for z in range(256) ]

def solve_svd(A,b):
    # compute svd of A
    U,s,Vh = np.linalg.svd(A,full_matrices=False)

    # U diag(s) Vh x = b <=> diag(s) Vh x = U.T b = c
    c = np.dot(U.T,b)
    # diag(s) Vh x = c <=> Vh x = diag(1/s) c = w (trivial inversion of a diagonal matrix)
    w = np.dot(np.diag(1/s),c)
    # Vh x = w <=> x = Vh.H w (where .H stands for hermitian = conjugate transpose)
    x = np.dot(Vh.conj().T,w)
    
    return x
def lnE(Zij,n):
    sumup = 0
    sumdown = 0
    for i in range(Zij.shape[0]):  
        w = Weight[Zij[i]]   
        sumup += w*(g[Zij[i],n]-B[i])
        sumdown += w
    
    lnEi = sumup/sumdown
    
    return np.exp(lnEi)

def draw_radiance_map(img_list,n):
    img_tmp =np.array(img_list)
    # print(img_tmp.shape)
    out = np.zeros((img_tmp.shape[1],img_tmp.shape[2]))
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            out[i,j] = lnE(img_tmp[:,i,j,n],n)
            if out[i,j] >1000:
                print(i,j,out[i,j])
    # print(np.sort(out,axis=1)[0][-5:])
    # print(out)
    #out = (out/np.max(out))*255
    # print(np.max(out),np.min(out))
    plt.imshow(out.astype(int), cmap="jet", origin="lower")
    plt.colorbar()
    plt.show()
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

    A[l,128,:] = 1
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
    draw_radiance_map(img_list,0)
    # lnE = np.array([x[i,:] for i in range(256,x.shape[0])])

    plt.plot(g[:,0]) 
    plt.show()

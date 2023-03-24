import numpy as np
class Robertson():
    def __init__(self,Zij,B,weight):
        self.Zij = Zij
        self.weight = weight
        self.t = B
        self.E = np.zeros((self.Zij.shape[0],self.Zij.shape[2]),dtype=np.float32)
        self.G = np.array([np.exp(np.arange(0, 1, 1 / 256))] * 3,dtype=np.float32).reshape((256,3))
       
    def getE(self):
        # print(self.G.shape)
        for k in range(self.Zij.shape[2]):
            for i in range(self.Zij.shape[0]):
                top = 0
                down = 0
                for j in range(self.Zij.shape[1]):
                    t = self.t[j]
                    z = self.Zij[i,j,k]
                    w = self.weight[z]
                    top += w*self.G[z,k]*t
                    down += w*t**2
                self.E[i,k] = top/down
    def getG(self):
        
        for c in range(3):
            z = self.Zij[...,c]
            E = self.E[...,c]
            for i in range(256):
                idx = np.where(z == i)
                # print(idx[1])
                w = self.weight[z[idx]]
                top = np.sum(w*E[idx[0]]*self.t[idx[1]])
                down = np.sum(w)
                if down >0:
                    self.G[i,c] = top/down

    def solve(self,epochs=15):
        for e in range(epochs):
            print('epochs:',e)
            
            self.getE()
            self.getG()
        return self.G
    
class Debvec():
    def __init__(self,Zij,B,Weight,N,P=256):
        self.A = np.zeros((N*P+254+1,256+P,3),dtype=np.float64)
        self.b = np.zeros(( self.A.shape[0],1,3),dtype=np.float64)
        self.x = np.zeros((256+P,3),dtype=np.float64)
        self.B = B
        self.Weight = Weight
        self.Zij = Zij
    def solve_svd(aelf,A,b):
   
        inv_A = np.linalg.pinv(A)
        x = np.dot(inv_A, b)
        return x.astype(np.float32)
    
    def solve(self):
        l=0
        for i in range(self.Zij.shape[0]):
            for j in range(self.Zij.shape[1]):
                for k in range(self.Zij.shape[2]):
                    Wij = self.Weight[self.Zij[i,j,k]]
                    self.A[l,self.Zij[i,j,k],k] = Wij
                    self.A[l,256+i,k] = -Wij
                    self.b[l,0,k] = Wij*self.B[j]
                l += 1

        self.A[l,127,:] = 1
        l += 1
        for i in range(0,254):
            self.A[l,i,:] = self.Weight[i+1]
            self.A[l,i+1,:] = -2*self.Weight[i+1]
            self.A[l,i+2,:] = self.Weight[i+1]
            l += 1
        self.x[:,0] = self.solve_svd( self.A[:,:,0],self.b[:,:,0]).squeeze()
        self.x[:,1] = self.solve_svd( self.A[:,:,1],self.b[:,:,1]).squeeze()
        self.x[:,2] = self.solve_svd( self.A[:,:,2],self.b[:,:,2]).squeeze()
        g = self.x[:256,:]
        return g
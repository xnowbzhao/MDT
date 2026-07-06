import os
import numpy as np
np.set_printoptions(precision=3, suppress=True)
def normalize(v):
    norm = np.linalg.norm(v)
    if norm == 0: 
       return v
    return v / norm
def rotation_matrix_around_x(deg):
    return np.array(np.matrix([[1,  0, 0], 
                     [0, np.cos(deg), -np.sin(deg)],
                     [0, np.sin(deg), np.cos(deg)]]))

def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    if any(v): #if not all zeros then 
        c = np.dot(a, b)
        s = np.linalg.norm(v)
        kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
        return np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    else:
        return np.eye(3) #cross of all zeros only occurs on identical directions

class Loader():
    def __init__(self, dataset_folder, batchsize, length, ncase, lsdsize=20, patchsize=80, rn=True):
        self.totalcase=ncase
        qurry=[]
        self.lsd=[]
        self.gt=[]
        self.xyz=[]
        self.patch=[]
        self.dr=[]
        self.vp=[]
        self.gtvp=[]
        self.patchsize=patchsize
        self.lsdsize=lsdsize
        self.rn=rn
        datapath=list(dataset_folder)
        for i in range(length):
            datapath[-9]=str(int(i/10))[0]
            datapath[-8]=str(int(i%10))[0]
            datapath[-6]='0'
            datapath[-5]='0'
            print("".join(datapath))
            self.lsd.append(np.fromfile("".join(datapath), dtype=np.float32))
            datapath[-6]='1'
            self.gt.append(np.fromfile("".join(datapath), dtype=np.float32))
            datapath[-6]='2'
            self.xyz.append(np.fromfile("".join(datapath), dtype=np.float32))
            datapath[-6]='3'
            self.patch.append(np.fromfile("".join(datapath), dtype=np.int32))
            datapath[-6]='4'
            self.dr.append(np.fromfile("".join(datapath), dtype=np.float32))
            datapath[-6]='5'
            self.vp.append(np.fromfile("".join(datapath), dtype=np.float32))
            datapath[-6]='6'
            self.gtvp.append(np.fromfile("".join(datapath), dtype=np.float32))
        for i in range(length):     
            self.lsd[i]=self.lsd[i].reshape((-1,lsdsize,lsdsize,3))
            self.gt[i]=self.gt[i].reshape((-1,3))
            self.xyz[i]=self.xyz[i].reshape((-1,3))
            self.patch[i]=self.patch[i].reshape((-1,patchsize))
            self.dr[i]=self.dr[i].reshape((-1,3))
            self.vp[i]=self.vp[i].reshape((-1,9))
            self.gtvp[i]=self.gtvp[i].reshape((-1,9))
            facenumber=self.lsd[i].shape[0]
            print(facenumber, self.lsd[i].shape, self.gt[i].shape, self.xyz[i].shape, self.patch[i].shape,self.dr[i].shape, self.vp[i].shape,self.gtvp[i].shape)
            temparray=np.zeros((facenumber,2)).astype(int)
            temparray[:,0]=i
            temparray[:,1]=range(0,facenumber)
            #print(temparray)
            qurry.append(temparray)
            
        self.datalist=np.vstack(qurry)
        np.random.shuffle(self.datalist)
        print(self.datalist.shape)
        if ncase!=-1:
            self.datalist=self.datalist[0:ncase]
        else:    
            self.totalcase=self.datalist.shape[0]     
        print(self.datalist.shape)
            
            
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.totalcase
        
    def __getitem__(self, idx):
        meshidx=self.datalist[idx][0]
        faceidx=self.datalist[idx][1]
        #print(meshidx,faceidx)
        idxlist= self.patch[meshidx][faceidx]
        ret= {'lsd': self.lsd[meshidx][idxlist],
        'gt': self.gt[meshidx][idxlist],
        'dr': self.dr[meshidx][idxlist],
        'xyz': self.xyz[meshidx][idxlist],
        'vp': self.vp[meshidx][idxlist],
        'gtvp': self.gtvp[meshidx][idxlist]
        }



        ret['vp']=ret['vp'].reshape((-1,3))
        ret['gtvp']=ret['gtvp'].reshape((-1,3))

        zeropoint=ret['xyz'][0,:].copy()
        ret['xyz']-= zeropoint
        ret['vp']-= zeropoint
        ret['gtvp']-= zeropoint

        avgn=np.zeros(3)
        for i in range(self.patchsize):
            avgn+=ret['lsd'][i][self.lsdsize//2][self.lsdsize//2]

        Tn=np.zeros(3)
        if self.rn is True:
            Tn[0]=1+np.random.rand()*0.4-0.2
            Tn[1]=np.random.rand()*0.4-0.2
            Tn[2]=np.random.rand()*0.4-0.2
            Tn=normalize(Tn)
        else:
            Tn[0]=1

        mat=rotation_matrix_from_vectors(avgn,Tn)

        if self.rn is True:
            Xn=np.deg2rad(np.random.rand()*360)
            mat2=rotation_matrix_around_x(Xn)
        else:
            mat2=np.eye(3)

        ret['lsd']=np.expand_dims(ret['lsd'],3)
        ret['lsd']=np.squeeze(np.matmul(np.matmul(ret['lsd'], mat.T), mat2.T))

        ret['gt']=np.expand_dims(ret['gt'],1)
        ret['gt']=np.squeeze(np.matmul(np.matmul(ret['gt'], mat.T), mat2.T))

        ret['dr']=np.expand_dims(ret['dr'],1)
        ret['dr']=np.squeeze(np.matmul(np.matmul(ret['dr'], mat.T), mat2.T))

        ret['xyz']=np.expand_dims(ret['xyz'],1)
        ret['xyz']=np.squeeze(np.matmul(np.matmul(ret['xyz'], mat.T), mat2.T))

        ret['vp']=np.expand_dims(ret['vp'],1)
        ret['vp']=np.squeeze(np.matmul(np.matmul(ret['vp'], mat.T), mat2.T))

        ret['gtvp']=np.expand_dims(ret['gtvp'],1)
        ret['gtvp']=np.squeeze(np.matmul(np.matmul(ret['gtvp'], mat.T), mat2.T))

        tempmax= np.max( [np.max(np.absolute(ret['xyz'])), np.max(np.absolute(ret['vp'])), np.max(np.absolute(ret['gtvp']))])
 

        ret['xyz']/=tempmax
        ret['vp']/=tempmax
        ret['gtvp']/=tempmax

        ret['gtvp']=ret['gtvp']-ret['vp']
        ret['vp']=ret['vp'].reshape((-1,9))
        ret['gtvp']=ret['gtvp'].reshape((-1,9))

        return ret





class Loader2():
    def __init__(self, lsdsize=20, patchsize=80):

        self.lsd=np.fromfile("lsd.bin", dtype=np.float32)
        self.xyz=np.fromfile("xyz.bin", dtype=np.float32)
        self.patch=np.fromfile("index.bin", dtype=np.int32)
        self.dr=np.fromfile("dr.bin", dtype=np.float32)
        self.vp=np.fromfile("vp.bin", dtype=np.float32)

        self.lsd=self.lsd.reshape((-1,lsdsize,lsdsize,3))
        self.xyz=self.xyz.reshape((-1,3))
        self.patch=self.patch.reshape((-1,patchsize))
        self.dr=self.dr.reshape((-1,3))
        self.vp=self.vp.reshape((-1,9))

        self.totalcase=self.patch.shape[0]
                          
        self.patchsize=patchsize
        self.lsdsize=lsdsize
            
            
    def __len__(self):
        ''' Returns the length of the dataset.
        '''
        return self.totalcase
        
    def __getitem__(self, idx):

        idxlist= self.patch[idx]
        #print(idxlist)
        ret= {'lsd': self.lsd[idxlist],
        'xyz': self.xyz[idxlist],
        'dr': self.dr[idxlist],
        'rmat': np.zeros((3,3)),
        'vp': self.vp[idxlist]
        }


        ret['vp']=ret['vp'].reshape((-1,3))
        zeropoint=ret['xyz'][0,:].copy()
        ret['xyz']-= zeropoint
        ret['vp']-= zeropoint


        avgn=np.zeros(3)
        for i in range(self.patchsize):
            avgn+=ret['lsd'][i][self.lsdsize//2][self.lsdsize//2]
 
        mat=rotation_matrix_from_vectors(avgn,[1,0,0])

        ret['rmat']=rotation_matrix_from_vectors([1,0,0],avgn)
        ret['zeropoint']=zeropoint


        ret['lsd']=np.expand_dims(ret['lsd'],3)
        ret['lsd']=np.squeeze(np.matmul(ret['lsd'], mat.T))

        ret['dr']=np.expand_dims(ret['dr'],1)
        ret['dr']=np.squeeze(np.matmul(ret['dr'], mat.T))

        ret['xyz']=np.expand_dims(ret['xyz'],1)
        ret['xyz']=np.squeeze(np.matmul(ret['xyz'], mat.T))

        ret['vp']=np.expand_dims(ret['vp'],1)
        ret['vp']=np.squeeze(np.matmul(ret['vp'], mat.T))

        tempmax= np.max( [np.max(np.absolute(ret['xyz'])), np.max(np.absolute(ret['vp']))])
        ret['scale']=tempmax

        ret['xyz']/=tempmax
        ret['vp']/=tempmax

        ret['vp']=ret['vp'].reshape((-1,9))


        return ret





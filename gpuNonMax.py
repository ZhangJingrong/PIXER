import pycuda.driver as drv
import pycuda.tools
import pycuda.autoinit
import numpy as np
import numpy.linalg as la
from pycuda.compiler import SourceModule
import Image, ImageDraw

import cv2
import sys, os,getopt 
from mpi4py  import MPI

import math, random
#from itertools import product
#from ufarray import *


from  EMAN2 import *

mod = SourceModule("""
__global__ void scoreGpu(float* heat,float*gaus, float*score, int sizex, int sizey, int psize)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;

    int tmp=psize/2;
    if((tx >= tmp) && (ty >= tmp) && (tx <= sizex-tmp) && (ty < sizey-tmp) ){
        int sx=tx-tmp;
        int sy=ty-tmp;
        heat=heat+sx*sizex+sy;
        //heat=heat+tx*sizex+ty;
        float sum=0;
        int sub=0;
        for(int i=0;i<psize;i++){
            for(int j=0;j<psize;j++){
                sum=sum+gaus[sub]*heat[j];
                sub=sub+1;
            }
            heat=heat+sizex;
        }
        score[tx*sizex+ty]=sum;

    }//end if 
}
__global__ void getMax(float* score, float* list, int size, int sizex, int sizey, int numx){
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;
    int cx=tx*size;
    int cy=ty*size;


    if(cx<sizex&&cy<sizey){
        int sx=size;
        int sy=size;

        if(cx+size>sizex){
            sx=sizex-cx;
        }
        if(cy+size>sizey){
            sy=sizey-cy;
        }
        score=score+cx*sizex+cy;
        //float max=6.805646932770577*(1000000000000000000000);
        float max=0;
        int maxx=0;
        int maxy=0;
        for(int i=0;i<sx;i++){
            for(int j=0;j<sy;j++){
                if(score[j]>=max){
                    max=score[j];
                    maxx=cx+i;
                    maxy=cy+j;
                }
            }
            score=score+sizex;
        }
    
        int sub=5*(tx*numx+ty);
        list[sub]=maxx;
        list[sub+1]=maxy;
        list[sub+2]=max;
        list[sub+3]=tx;
        list[sub+4]=ty;
    }
}

__global__ void getMax3(float* score, float* list, int psize, int sizex, int sizey, int numx,int num, int iter){
    int tx=threadIdx.x;
    int ty=threadIdx.y;

    tx=blockIdx.x*blockDim.x+tx;
    ty=blockIdx.y*blockDim.y+ty;
    int sub=(tx*numx+ty)*5;

    if (sub<num){
        int cx=list[sub]-psize/2;
        int cy=list[sub+1]-psize/2;

        float max=0;
        int maxx=0;
        int maxy=0;

        for(int i=0;i<iter;i++){

            if(cx<sizex&&cy<sizey&&cx>0&&cy>0){
                int sx=psize;
                int sy=psize;

                if(cx+sx>sizex){
                    sx=sizex-cx;
                }
                if(cy+sy>sizey){
                    sy=sizey-cy;
                }
                max=0;
                maxx=0;
                maxy=0;
                float* score0=score+cx*sizex+cy;
                for(int i=0;i<sx;i++){
                    for(int j=0;j<sy;j++){
                        if(score0[j]>=max){
                            max=score0[j];
                            maxx=cx+i;
                            maxy=cy+j;
                        }
                    }
                    score0=score0+sizex;
                }
                cx=maxx-psize/2;
                cy=maxy-psize/2;
            }
         }//end for

        list[sub]=maxx;
        list[sub+1]=maxy;
        list[sub+2]=max;
        list[sub+3]=tx;
        list[sub+4]=ty;
    }//end if num
}



__global__ void multiply_them(float *dest, float *a, float *b)
{
  const int i = threadIdx.x;
  dest[i] = a[i] * b[i];
}

""")

class item:
    def __init__(self,count=0,x=0 ,y=0 ):
        self.id=count;
        self.minX = x;
        self.minY=y
        self.maxX=x;
        self.maxY=y;
        self.totalN=0;
        self.totalS=0;
        self.x=x;
        self.y=y;
        self.p=0.1;
    def __str__(self):
        line=str(self.id)+' '+str(self.minX)+' '+str(self.maxX)+' '+str(self.minY)+' '+str(self.maxY) \
                +' '+str(self.totalN)+' '+str(self.totalS)+' '+str(self.x)+' '+str(self.y)+' '+str(self.p)+'\n';
        return line;

    def update(self,x,y):
        if(x<self.minX):
            self.minX=x;
        elif(x>self.maxX):
            self.maxX=x;

        if(y<self.minY):
            self.minY=y;
        elif(y>self.maxY):
            self.maxY=y;
        
        self.totalN=self.totalN+1;

        return self;

    def getS(self):
        width=self.maxX-self.minX+1;
        length=self.maxY-self.minY+1;
        self.totalS=width*length;
        self.x=self.minX+width/2;
        self.y=self.minY+width/2;
        self.p=float(self.totalN)/self.totalS;
        return self.totalS;





def binImg(GrayImage, psize):
    GrayImage= cv2.convertScaleAbs(GrayImage)
    GrayImage= cv2.medianBlur(GrayImage,5);
    GrayImage= cv2.medianBlur(GrayImage,5);
    if psize%2==0:
        psize=psize+1;
    thresh1,th2=cv2.threshold(GrayImage,200,255,cv2.THRESH_BINARY);

    #th2 = cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,psize,2)
    return th2;

def reshape(res,num):
    canList=[];
    for i in range(0,num):
        sub=i*5;
        m=res[sub+2];
        if m!=0:
            candi=[int(res[sub]),int(res[sub+1]),m,int(res[sub+2]),int(res[sub+3])];
            canList.append(candi);
    print('convert');
    canList.sort(lambda x,y:cmp(x[2],y[2]), reverse=True);

    #for i in range(0, 10):
    #    print(canList[i]);
    return canList;

def sBox(res, numx, numy,gSize):
    cBox=np.zeros([numx,numy]);
    xBox=np.zeros([numx,numy]).astype(float);
    yBox=np.zeros([numx,numy]).astype(float);
    scBox=np.zeros([numx,numy]).astype(float);

    nList=[];

    for r in res:
        mx,my,p,nx,ny=r;
        tadd=-1;
        s=mx%gSize;
        step=20;
        step2=gSize-step;

        if s>step and s<step2:
            tadd=0;
        elif s>step2:
            tadd=1;

        tx=mx/gSize+tadd;
        if tx>= numx:
            tx=numx-1;
        
        tadd=-1;
        s=my%gSize;

        if s>step and s<step2:
            tadd=0;
        elif s>step2:
            tadd=1;
        ty=my/gSize+tadd;

        if ty>= numy:
            ty=numy-1;

        cBox[tx,ty]=cBox[tx,ty]+1;
        xBox[tx,ty]=xBox[tx,ty]+mx;
        yBox[tx,ty]=yBox[tx,ty]+my;
        scBox[tx,ty]=scBox[tx,ty]+p;



    for i in range(0, numx):
        for j in range(0, numy):
            n=cBox[i,j];
            if(n!=0):
                ix=int(xBox[i,j]/n);
                iy=int(yBox[i,j]/n);
                sc=scBox[i,j];
                res=[ix,iy,sc];
                nList.append(res);


    nList.sort(lambda x,y:cmp(x[2],y[2]), reverse=True);
    l=int(len(nList)*0.8);
    nList=nList[:l];

    return nList;


def scoreGpuLaunch(heat, gaus,psize,gsize, niter,fscore):
    heat=heat.astype(np.float32);
    gaus=gaus.astype(np.float32);
    score=np.zeros(heat.shape).astype(np.float32);
    [sizex,sizey]=heat.shape;
    sizex=np.int32(sizex);
    sizey=np.int32(sizey);
    psize=np.int32(psize);
    gsize=np.int32(gsize);
    
    func = mod.get_function("scoreGpu")
   
    tx=16;
    ty=16;
    bx=(sizex-1)/tx+1;
    by=(sizey-1)/ty+1;
    print('get score gpu',tx, ty, bx, by, gsize, psize);

    
    if os.path.exists(fscore)==0:
        func(drv.In(heat),drv.In(gaus), drv.Out(score),sizex, sizey, psize,
            block=(tx,ty,1),grid=(bx,by));
        b=EMNumPy.numpy2em(score);
        b.write_image(fscore, 0, IMAGE_MRC, False, None, EM_FLOAT);
    else:
        em=EMData(fscore);
        score=EMNumPy.em2numpy(em);
        print(score.shape);
    
    numx=int((sizex-1)/gsize+1);
    numy=int((sizey-1)/gsize+1);
    num=int(numx*numy);
    numx=np.int32(numx);
    numy=np.int32(numy);
    res=np.zeros(num*5).astype(np.float32);
    tnum=np.int32(num*5);
    tx=16;
    ty=16;
    bx=int((numx-1)/tx+1);
    by=int((numy-1)/ty+1);
    
    print('get Max',tx, ty, bx, by);

    func = mod.get_function("getMax")
    func(drv.In(score),drv.Out(res),gsize,sizex, sizey, numx,
        block=(16,16,1),grid=(bx,by,1));
    
    niter=np.int32(niter);
    niterTest=np.int32(1);
    func = mod.get_function("getMax3");
    gsize=np.int32(gsize);

    print('get Max3',tx, ty, bx, by,niter,'other : ', gsize, sizex, sizey, numx, tnum);
    #for i in range(0, niter):
    func(drv.In(score),drv.InOut(res),gsize,sizex, sizey, numx, tnum,niter,
            block=(16,16,1),grid=(bx,by,1));

    print('reshape');
    res=reshape(res,num);
    return res,score;

def gaussian_kernel_2d_opencv(kernel_size = 3,sigma = 1):
    kx = cv2.getGaussianKernel(kernel_size,sigma)
    ky = cv2.getGaussianKernel(kernel_size,sigma)
    res=np.multiply(kx,np.transpose(ky))
    res=1-(res-np.max(res))/(-1*np.ptp(res));
    return res;


def heatToScore(heat,psize,gaus):
    score=np.zeros(heat.shape);
    [sizex,sizey]=heat.shape;
    b=int(psize/2);
    sx=sy=b;
    ex=sizex-b;
    ey=sizey-b;

 
    for x in range(sx,ex):
        for y in range(sy,ey):
            a=heat[x-b:x+b,y-b:y+b]
            a=a*gaus;
            score[x,y]=np.sum(a);

    return score;


def genCandidate(score,psize,pCan):
    [sizex,sizey]=score.shape;
    tmp=psize/2;
    gap=int(psize*pCan);
    sx=sy=tmp;
    ex=sizex-gap;
    ey=sizey-gap;
    canList=[];
    for x in range(sx,ex,gap):
        for y in range(sy,ey,gap):
            a=score[x:x+gap,y:y+gap];
            vmax=score[x,y];
            v=np.unravel_index(np.argmax(a, axis=None), a.shape);
            ix=int(v[0]+x);
            iy=int(v[1]+y);
            candi=[ix,iy,a[v[0],v[1]]];
            canList.append(candi);

    return canList;


def sorList(l, im):
    l.sort(lambda x,y:cmp(x[2],y[2]), reverse=True);

    show=10;
    return l
def nonMax2(heatArr, bArr, psize, pnum,starFile, pCan,op1):
    gaus=gaussian_kernel_2d_opencv(kernel_size = psize,sigma = 0);
    score=heatToScore(heatArr, psize,gaus);
    canList=genCandidate(score,psize,pCan);
    canList=sorList(canList,heatArr);
    if(len(canList)>pnum):
        canList=canList[:pnum];
    return canList;

def nonMax(heatArr, psize, pnum,starFile, pCan,niter, fscore):
    gaus=gaussian_kernel_2d_opencv(kernel_size = psize,sigma = 0);
    gsize=psize*pCan;
    print('gsize:', gsize,psize, 'iter:', niter);
    res,score=scoreGpuLaunch(heatArr,gaus,psize, gsize, niter,fscore);
    return res,score;

def cleanCanList(canList,op1,psize):
    numCan=len(canList);
    op=op1*psize*psize;
    tmp=[-psize,-psize,0];
    for i in range(0,len(canList)):
        candi=canList[i];
        if candi !=tmp:
            x=candi[0];
            y=candi[1];

            for j in range(i+1,len(canList)) :
                candi=canList[j];
                rx=candi[0];
                ry=candi[1];
                
                if abs(x-rx)<psize and abs(y-ry)<psize:
                    sOp=getOverlap(x,y,rx,ry,psize);
                    if sOp>op:
                        #print(rx,ry,sOp,op)
                        canList[j]=tmp;
                        (canList[i])[0]=0.8*x+0.2*rx;
                        (canList[i])[1]=0.8*y+0.2*ry;

def cleanCanList(canList,op1,psize):
    numCan=len(canList);
    op=op1*psize*psize;
    tmp=[-psize,-psize,0];
    for i in range(0,len(canList)):
        candi=canList[i];
        if candi !=tmp:
            x=candi[0];
            y=candi[1];

            for j in range(i+1,len(canList)) :
                candi=canList[j];
                rx=candi[0];
                ry=candi[1];
                
                if abs(x-rx)<psize and abs(y-ry)<psize:
                    sOp=getOverlap(x,y,rx,ry,psize);
                    if sOp>op:
                        #print(rx,ry,sOp,op)
                        canList[j]=tmp;
                        (canList[i])[0]=0.8*x+0.2*rx;
                        (canList[i])[1]=0.8*y+0.2*ry;


    return [x for x in canList if x != tmp]

def cleanBinList(canList, bArr,psize):
    tmp=[-psize,-psize,0];
    for i in range(0,len(canList)):
        l=canList[i];
        x=l[0];
        y=l[1];
        if bArr[x,y]==0:
            canList[i]=tmp;

    return [x for x in canList if x != tmp];

def getOverlap(x,y,rx,ry,p):
    t=int(p/2);
    xmin=x-t
    xmax=x+t
    ymin=y-t
    ymax=y+t

    rxmin=rx-t;
    rxmax=rx+t;
    rymin=ry-t;
    rymax=ry+t;

    x1=abs(rxmin-xmax);
    x2=abs(rxmax-xmin);
    lx=min(x1,x2);

    y1=abs(rymin-ymax);
    y2=abs(rymax-ymin);
    ly=min(y1,y2);

    return lx*ly;

def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    star.write('_rlnParticleSelectZScore  #3\n');
    return star 

def writeCan(fstar,canList,sizex, sizey):
    fstar=writeStarHead(fstar);
    p=0;
    l=len(canList);
    for i in range(0,l):
        c=canList[i];
        #line=str(c[0])+' '+str(c[1])+' '+str(c[2])+'\n';
        #x=c[1];
        #y=sizey-c[0];
        x=c[0];
        y=c[1];
        x0=y;
        y0=x;
        if(len(c)>2):
            p=c[2];

        line=str(x0)+' '+str(y0)+' '+str(p)+'\n';
        #print(line);
        fstar.write(line);
    fstar.close();

def getList(label,psize,fin ,fout):
    total=psize*psize;
    cList={}
    count=0;
    for (x, y) in label:
        iden=label[(x, y)];
        if(cList.has_key(iden)==0):
            count=count+1;
            a=item(iden,x,y);
            cList[iden]=a;
        else:
            cList[iden].update(x,y);

    for iden in cList:
        cList[iden].getS();
    return cList;

def adCor(conList, resList,labels, bArr,psize):
    t=psize/2;
    for i in range(0, len(resList)):
        res=resList[i];
        x,y,p=res;
        ocount=np.sum(bArr[x-t:x+t, y-t:y+t]);

        if labels.has_key((x,y)):
            iden=labels[(x,y)];
            c=conList[iden];
            minx=c.minX;
            miny=c.minY;
            maxx=c.maxX;
            maxy=c.maxY;

            cx=minx+((x-minx)/psize)*psize+t;
            cy=minx+((y-miny)/psize)*psize+t;
            dx=maxx-((maxx-x)/psize)*psize-t;
            dy=maxy-((maxy-y)/psize)*psize-t;
            
            ccx=(cx+dx)/2;
            ccy=(cy+dy)/2;

            count=np.sum(bArr[ccx-t:ccx+t, ccy-t:ccy+t]);
            if ocount<count*1.2 :
                (resList[i])[0]=ccx;
                (resList[i])[1]=ccy;
                print('from: ', x,y, 'to :', cx, cy,' to: ', dx,dy, 'final: ', ccx,ccy);
        else:
                (resList[i])[0]=0;
                (resList[i])[1]=0;
           
    return resList; 

def run(img):
    data = img.load()
    width, height = img.size
    # Union find data structure
    uf = UFarray()
    labels = {}
    vPass=0;
    vUse=255;
    for y, x in product(range(height), range(width)):
        if data[x, y] == vPass:
            pass
        elif y > 0 and data[x, y-1] == vUse:
            labels[x, y] = labels[(x, y-1)]
        elif x+1 < width and y > 0 and data[x+1, y-1] == vUse:
            c = labels[(x+1, y-1)]
            labels[x, y] = c
            if x > 0 and data[x-1, y-1] == vUse:
                a = labels[(x-1, y-1)]
                uf.union(c, a)
            elif x > 0 and data[x-1, y] == vUse:
                d = labels[(x-1, y)]
                uf.union(c, d)
        elif x > 0 and y > 0 and data[x-1, y-1] == vUse:
            labels[x, y] = labels[(x-1, y-1)]
        elif x > 0 and data[x-1, y] == vUse:
            labels[x, y] = labels[(x-1, y)]
        else: 
            labels[x, y] = uf.makeLabel()
    uf.flatten()
    colors = {}
    output_img = Image.new("RGB", (width, height))
    outdata = output_img.load()
    for (x, y) in labels:
        component = uf.find(labels[(x, y)])
        labels[(x, y)] = component
        if component not in colors: 
            colors[component] = (random.randint(0,255), random.randint(0,255),random.randint(0,255))
        outdata[x, y] = colors[component]
    return (labels, output_img)

def reHeat(heatArr, psize,ratio):
    sizex,sizey=heatArr.shape;
    sizex=int(sizex/ratio);
    sizey=int(sizey/ratio);
    psize=int(psize/ratio);

    heatArr=cv2.resize(heatArr,(sizex, sizey),interpolation=cv2.INTER_CUBIC)
    return heatArr, psize;


def reCan(canList, ratio):
    for i in range(0,len(canList)):
        x=canList[i][0];
        y=canList[i][1];

        canList[i][0]=x*ratio;
        canList[i][1]=y*ratio;
    return canList;

def reNorm(heatArr,psize, nSep):
    ksize=3;
    
    kernel = np.ones((ksize,ksize),np.uint8)
    img = cv2.erode(heatArr,kernel,iterations = nSep)
    #tmp=int(psize/4);
    #if tmp%2 ==0:
    #    tmp=tmp+1;
    #heatArr= cv2.medianBlur(heatArr,tmp);
    #heatArr= cv2.medianBlur(heatArr,5);
    #heatArr= cv2.medianBlur(heatArr,5);
    #heatArr= cv2.medianBlur(heatArr,5);
    return heatArr;

def reLev(heatArr, level):
    num=int(255/level);
    heatArr=heatArr/num;
    heatArr=heatArr.astype(int);
    return heatArr;

def draw(im,cx,cy,sx,sy):
    startx=cx-sx/2;
    starty=cy-sx/2;
    endx=cx+sx/2;
    endy=cy+sx/2;

    p=1;
   
    c=1;
    r=c/4;
    g=(c-r*4)/2;
    b=c%2;
    r=r*255;
    g=g*255;
    b=b*255;

    w=2;
    
    cv2.rectangle(im, (startx, starty), (endx, endy), (b,g,r), w)

    return im;

def checkScore(score, fstar):
    score0=np.zeros([score.shape[0], score.shape[1],3]);
    print(score0.shape, score.shape);
    score0[:,:,0]=score[:,:];
    score0[:,:,1]=score[:,:];
    score0[:,:,2]=score[:,:];
    a=score0;
    score0=(255*(a - np.max(a))/-np.ptp(a))
    for c in canList:
        x=int(c[0]);
        y=int(c[1]);
        sx=sy=psize;
        score0=draw(score0,x,y,sx,sy);

    cv2.imwrite(fstar[:-4]+'.jpg',score0);

def processOne(fin,fout,fstar, fheat,psize,pCan,overlap,fscore, nIter, nSep):
    heatArr=cv2.imread(fheat,0);
    ratio=1;
    pnum=100;
    level=3;
    heatArr=reNorm(heatArr,psize, nSep);
    heatArr=reLev(heatArr,level);
    canList,score=nonMax(heatArr, psize, pnum, fstar,pCan,nIter, fscore);
    print('clean LIST');
    canList=cleanCanList(canList,overlap,psize);
    #canList=reCan(canList,ratio);
    print('Number of Particles:', len(canList));
    sizex,sizey=heatArr.shape;
    writeCan(fstar,canList,sizex,sizey);

if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:r:s:p:z:n:g:t:x:dhc") 
    din="" 
    dout=""
    dstar=""
    pSize=0;
    isDir=0;
    pCan=0.1;
    overlap=0.6
    num=-1;
    dheat="";
    nIter=100;
    nSep=10;
    for op, value in opts: 
        if op == "-i": 
            din = value 
        elif op == "-o": 
            dout = value
        elif op =="-r":
            dheat=value;
        elif op =="-s":
            dstar= value;
        elif op =="-p":
            pSize=int(value)
        elif op =="-t":
            nIter=int(value)
        elif op =="-g":
            pCan=float(value)
        elif op =="-x":
            nSep=int(value)
        elif op =="-z":
            overlap=float(value)
        elif op =="-n":
            num=int(value);
        elif op=="-d":
            isDir=1
        elif op == "-h": 
            usage() 
            sys.exit()
        elif op=="-c":
            din="/home/ict/git/python/pickEye/testDataset/pdb1f07/bin/pdb1f07-29.png";
            dout="/home/ict/git/python/pickEye/testDataset/pdb1f07/star/pdb1f07-29.png";
            dstar="/home/ict/git/python/pickEye/testDataset/pdb1f07/star/pdb1f07-29.star";
            pSize=110;
            dheat="/home/ict/git/python/pickEye/testDataset/pdb1f07/heat/pdb1f07-29.png";



    print(din,dout,dstar,pSize);

    if isDir ==1:
        comm=MPI.COMM_WORLD
        crank=comm.Get_rank();
        csize=comm.Get_size();

        if crank==0:
            if os.path.isdir(dout)==0:
                os.mkdir(dout);
        comm.barrier();
        count=0;

        fins=os.listdir(din);
        for i in range(crank,len(fins),csize):
            f=fins[i];
            if num>0 and count>num:
                break;
            if f[-3:]=='png':
                fin=os.path.join(din,f);
                
                fstar=f[:-4]+'.star';
                fstar=os.path.join(dstar,fstar);

                fout=os.path.join(dout,f);
                fheat=os.path.join(dheat,f);
                
                fscore=fout[:-3]+'mrc';
                
                print('Now processing:', fin,fstar,fout, fscore);
                processOne(fin,fout,fstar,fheat,pSize,pCan,overlap, fscore, nIter, nSep)

        comm.barrier();
    else:
        #if  os.path.isdir(dout)==0:
        #    os.mkdir(dout);

        fid=(din.split('/'))[-1];
        fheat=dheat;
        fout=dout;
        fstar=dstar;
        fscore=fout[:-3]+'.mrc';

        fid=fid[:-4]+'.star';
        
        print(din, fheat, fout, fstar);
        processOne(din,fout,fstar,fheat,pSize,pCan,overlap, fscore, nIter, nSep);

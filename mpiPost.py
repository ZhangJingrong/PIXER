import scipy.io as sio
import cv2
import numpy as np
import os,sys,getopt
import timeit

from mpi4py  import MPI
from pycuda.compiler import SourceModule
import pycuda.driver as drv
# -- initialize the device
import pycuda.autoinit
from EMAN2 import *

mod = SourceModule("""
__global__ void scoreGpu(int* heat, int* score, int sizex, int sizey, int sizep)
{
    // 2D Thread ID (assuming that only *one* block will be executed)
    int sx=blockIdx.x*blockDim.x+threadIdx.x;
    int sy=blockIdx.y*blockDim.x+threadIdx.y;
    int tmp=sizep/2;
    if((sx >= tmp) && (sy >= tmp) && (sx <= sizex-tmp) && (sy < sizey-tmp) ){
        int ex=sx+sizep;
        int ey=sy+sizep;
        for(int x=sx-tmp;x<ex;x++){
            for(int y=sy-tmp;y<ey;y++){
                score[sx,sy]+=heat[x*sizey+y];
            }//end for y
        }//end for x
    }//end if 
}
__global__ void scoreGpu2(int sizep){
     sizep=sizep+1;
}
""")


def scoreGpuLaunch(heat,sizep):
    score=np.zeros(heat.shape);

    #h_gpu = cuda.mem_alloc(heat.nbytes);
    #cuda.memcpy_htod(h_gpu, heat);

    [sizex,sizey]=heat.shape;
    sizex=np.int32(sizex);
    sizey=np.int32(sizey);
    sizep=np.int32(sizep);

    func = mod.get_function("scoreGpu2")
    nT=int(128);
    nX=int((sizex-1)/nT+1);
    nY=int((sizey-1)/nT+1);

    #print(nT,nX,nY);
    #func(drv.In(heat),drv.Out(score),heat.shape[0], heat.shape[1], sizep, block=(128,128,1),grid=(8,8,1));
    func(sizep, block=(2,2,1),grid=(2,1,1));   
    return score;

def checkArr(s1,s2):
    [sizex,sizey]=s1.shape;
    for x in range(0,sizex):
        for y in range(0, sizey):
            if(s1[x,y]!=s2[x,y]):
                print('wrong!!!!!:',x,y);

def getId(fname):
    lines=fname.split('/');
    fname=lines[-1];
    lines=fname.split('.');

    size=int(lines[-2]);
    y=int(lines[-3]);
    x=int(lines[-4]);

    stop=-1*(len(lines[-1])+len(lines[-2])+len(lines[-3])+len(lines[-4])+4)

    fid=fname[:stop];
    #print(fid,x,y,size);
    return fid,x,y,size;

def getIdOri(fname):
    lines=fname.split('/');
    fname=lines[-1];
    lines=fname.split('.');
    stop=-1*(len(lines[-1])+1)
    fid=fname[:stop];
    return fid;

def readMat(fname,size):
    load_data = sio.loadmat(fname);
    load_matrix=load_data['data'];
    data=np.array(load_matrix);
    #data=data[:,:,1,0];
    data=data[:,:,0,0];    
    data = np.array(map(list,zip(*data)))
    data= cv2.resize(data, (size,size), interpolation=cv2.INTER_AREA);
    data=(255*(data - np.min(data))/np.ptp(data)).astype(np.int);
    return data;

def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    star.write('_rlnScore  #3\n');
    return star 

def binImg(GrayImage, psize):
    GrayImage= cv2.convertScaleAbs(GrayImage)
    GrayImage= cv2.medianBlur(GrayImage,5);
    GrayImage= cv2.medianBlur(GrayImage,5);
    GrayImage= cv2.medianBlur(GrayImage,5);
    if psize%2==0:
        psize=psize+1;
    th2 =cv2.adaptiveThreshold(GrayImage,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY,psize,2)
    return th2;

def heatToScore(heat,psize,starFile):
    score=np.zeros(heat.shape);
    [sizex,sizey]=heat.shape;
    b=int(psize/2);
    sx=sy=b;
    ex=sizex-b;
    ey=sizey-b;

    for x in range(sx,ex):
        for y in range(sy,ey):
            a=heat[x-b:x+b,y-b:y+b]
            score[x,y]=np.sum(a);

    return score;

def draw(im,l,starFile):
    sx=sy=10;
    count=0;
    for line in l:
        if count<30:
            cx,cy,p=line;
            startx=cx-sx/2;
            starty=cy-sx/2;
            endx=cx+sx/2;
            endy=cy+sx/2;
    
            #print('draw', cx,cy);

            for x in range(startx,endx):
                for y in range(starty,endy):
                    im[x,y]=0;
            fname=starFile[:-5]+'.png';
            cv2.imwrite(fname,im);
            count=count+1;



def genCandidate(score,psize,pCan):
    [sizex,sizey]=score.shape;
    tmp=psize/2;
    gap=int(psize*pCan);
    print(gap);
    sx=sy=tmp;
    ex=sizex-tmp-gap;
    ey=sizey-tmp-gap;
    canList=[];
    for x in range(sx,ex,gap):
        for y in range(sy,ey,gap):
            a=score[x:x+gap,y:y+gap];
            vmax=score[x,y];
            v=np.unravel_index(np.argmax(a, axis=None), a.shape);
            ix=int(v[0]+x);
            iy=int(v[1]+y);
            #print('2:',a[v[0],v[1]], ix,iy, gap);
            candi=[ix,iy,a[v[0],v[1]]];
            canList.append(candi);

    return canList;

def sorList(l, im):
    l.sort(lambda x,y:cmp(x[2],y[2]), reverse=True);
    #l.sort(lambda x,y:cmp(x[2],y[2]));

    show=10;

    for i in range(0, show):
        x,y,p=l[i];
        print(x,y,p);

    return l;
def overlap(x,y,rx,ry,p):
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

def cleanList(canList,op1,op2,psize):
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
                    sOp=overlap(x,y,rx,ry,psize);
                    if sOp>op:
                        canList[i]=tmp;
    return [x for x in canList if x != tmp]

def writeCan(fstar,canList,psize):
    l=len(canList);
    totalNum=psize*psize;
    for i in range(0,l):
        c=canList[i];
        p=float(255*totalNum-c[2])/254/totalNum;

        line=str(c[0])+' '+str(c[1])+' '+str(p)+'\n';
        fstar.write(line);
    fstar.close();


def nonMax(heatArr, psize, pnum,starFile, pCan,op1,op2):
    fstar=writeStarHead(starFile);
    s1 = timeit.timeit()

    score=heatToScore(heatArr, psize, starFile);
    canList=genCandidate(score,psize,pCan);

    canList=sorList(canList,heatArr);
    draw(heatArr,canList,starFile);
    canList=cleanList(canList,op1,op2,psize);
    if(len(canList)>pnum):
        canList=canList[:pnum];

    writeCan(fstar,canList,psize);


def writeHeat(dout,dbin,fid, img,weight, psize, pnum,dstar,pCan,op1,op2):
    f=fid+'.png';
    fname=os.path.join(dout,f);
    #print(np.argwhere(weight == 0));
    img=np.divide(img,weight);
    cv2.imwrite(fname,img);

    fbin=os.path.join(dbin,f);
    img=binImg(img,psize);
    cv2.imwrite(fbin,img);

def writeCompare(dout, fid, img):
    fname=fid+'.com.png';
    fname=os.path.join(dout,fname);
    cv2.imwrite(fname,img);

def getImgSize(dref,fid):
    fname=fid+'.png';
    fname=os.path.join(dref,fname);
    img=cv2.imread(fname,0);
    sizex=img.shape[0];
    sizey=img.shape[1];
    return sizex,sizey;

def mergeImg(din,dout,dbin,dref,flist,psize,pnum,dstar,crank,csize,pCan,op1,op2):
    flist=open(flist,'r');
    lines=flist.readlines();

    oldId=0;
    img=np.zeros([512,512]);
    weight=img;
    ref=weight;
    for f in lines:
        f=f[:-1];
        fid,x,y,size=getId(f);

        if oldId !=fid:
            if oldId!=0:
                writeHeat(dout,dbin,oldId,img,weight,psize,pnum,dstar,pCan,op1,op2);
            sizex,sizey=getImgSize(dref,fid);
            img=np.zeros([sizex,sizey]);
            weight=np.zeros([sizex,sizey]);
            oldId=fid;

        fname=f[:-4]+'.png_blob_0.mat';
        fname=os.path.join(din,fname);

        subImg=readMat(fname,size);

        if(x+size>img.shape[0]):
            x=img.shape[0]-512;
        if(y+size>img.shape[1]):
            y=img.shape[1]-512;

        img[x:x+size,y:y+size]=img[x:x+size,y:y+size]+subImg[:,:];
        weight[x:x+size,y:y+size]=weight[x:x+size,y:y+size]+1;

    if oldId!=0:
        writeHeat(dout,dbin, oldId, img,weight, psize, pnum,dstar,pCan,op1,op2);

def nonMaxDir(dout, psize,pnum, dstar, crank, csize,pCan,op1,op2):
    heatList=os.listdir(dout);
    for i in range(crank, len(heatList),csize):
        heatFile=heatList[i];
        fid=getIdOri(heatFile);
        heatFile=os.path.join(dout,heatFile);
        heatArr=cv2.imread(heatFile,0);

        starFile=fid+'.star';
        starFile=os.path.join(dstar,starFile);
        nonMax(heatArr, psize, pnum,starFile,pCan,op1,op2)
def usage():
    print("python postProcess.py -i din -o dout -r def -l flist -p psize -n pnum -s dstar -d ")


if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:b:r:l:p:n:s:dc:x:y:")
    din=""
    dout=""
    dref=""
    dbin=""
    flist=""
    isDir=0;
    psize=100;
    pnum=10;
    dstar="";
    isDir=1
    pCan=0.6;
    op1=0.7;
    op2=0.7;
    for op, value in opts:
        if op == "-i":
            din = value
        elif op == "-o":
            dout = value
        elif op == "-b":
            dbin=value
        elif op =="-r":
            dref= value
        elif op =="-l":
            flist= value
        elif op =="-p":
            psize=int(value);
        elif op =="-n":
            pnum=int(value);
        elif op =="-s":
            dstar=value
        elif op == "-h":
            usage()
            sys.exit()
        elif op =="-d":
            isDir=1;
        elif op =="-c":
            pCan= float(value)
        elif op =="-x":
            op1= float(value)
        elif op =="-y":
            op2= float(value);


    comm=MPI.COMM_WORLD
    crank=comm.Get_rank();
    csize=comm.Get_size();
    print('mpi',crank,'getPara:', din,dout,dbin,dref,flist,psize,pnum, dstar);
    flist=flist+'.'+str(crank);

    if crank==0:
        if os.path.isdir(dout)==0:
            os.mkdir(dout);
            print("creat dir:", dout);
        if os.path.isdir(dstar)==0:
            os.mkdir(dstar);
        if os.path.isdir(dbin)==0:
            os.mkdir(dbin);

    comm.barrier();

    mergeImg(din,dout,dbin,dref,flist,psize,pnum,dstar,crank, csize,pCan,op1,op2);
    
    comm.barrier();

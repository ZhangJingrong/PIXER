import cv2
import sys, os,getopt 
from mpi4py  import MPI
from  EMAN2 import *
import numpy as np
def norm(fin,n):
    em=EMData(fin);
    img=EMNumPy.em2numpy(em);
    arr2=normImg(img,n);
    return arr2;

def normImg(img,n):
    imax=np.max(img);
    imin=np.min(img);
    a=float(255)/(imax-imin);
    b=(-1)*a*imin;

    sizex=img.shape[0];
    sizey=img.shape[1];
    arr=np.zeros([sizex,sizey]);
    arr=np.round(a*img+b).astype(int);

    amax=np.max(arr);
    amin=np.min(arr);
    amean=np.mean(arr);
    astd=np.std(arr);

    bmin=amean-astd*n;
    bmax=amean+astd*n;
    c=float(255)/(bmax-bmin);
    d=(-1)*c*bmin;

    arr2=np.round(c*arr+d).astype(int);
    for x in range(0,sizex):
        for y in range(0,sizey):
            if arr2[x,y]<0:
                arr2[x,y]=0;
            elif arr2[x,y]>255:
                arr2[x,y]=255;

    arr3=np.zeros([sizex,sizey,3]);
    arr3[:,:,0]=arr2[:,:];
    arr3[:,:,1]=arr2[:,:];
    arr3[:,:,2]=arr2[:,:];

    return arr3;



def box(im,cx,cy, sx,sy,fout):

    sizex,sizey=im.shape;
    startx=cx-sx/2;
    starty=cy-sx/2;
    endx=cx+sx/2;
    endy=cy+sx/2;

    if startx<0:
        startx=0;
    if starty<0:
        starty=0;

    if endx>sizex:
        endx=sizex;
    if endy>sizey:
        endy=sizey;

    bArr=np.zeros([sx,sy]);

    bArr[0:endx-startx, 0: endy-starty]=im[startx:endx, starty:endy];
    print(fout,startx, endx, starty, endy, cx, cy);

    cv2.imwrite(fout, bArr);

def getCors(fstar):
    f=open(fstar,'r');
    plist=f.readlines();
    plist=plist[7:]
    return plist;

def boxOne(fid,fin,fstar,pSize,dout):
    isMrc=0;
    fList=[];

    if(fin[-3:]=='png'):
        im = cv2.imread(fin,0);
    elif (fin[-3:]=='mrc'):
        im=norm(fin,2);
        isMrc=1;

    plist=getCors(fstar);

    sx=pSize;
    sy=pSize;

    count=1;
    p=1;
    for line in plist:
        s=line.split(' ');
        x=float(s[0]);
        x=int(x);
        y=float(s[1]);
        y=int(y);
        if len(s)==3:
            p=float(s[2]);
        

        fout=fid+'.'+str(x)+'.'+str(y)+'.png';
        fout=os.path.join(dout,fout);
        
        if isMrc==0:
            cx=y;
            cy=x;
        else:
            cx=x;
            cy=y;

        box(im,cx,cy,sx,sy,fout);
        count=count+1
        
        fout=os.path.abspath(fout);
        fout=fout+'\n';
        fList.append(fout);
    return fList;


def usage():
    cm="mpiexec -n 6 -i ./KLH1/data -s ./KLH1/star -o ./KLH1/cpar -p 305 -d";
    print("mpiexec -n 6 -i ./KLH1/data -s ./KLH1/star -o ./KLH1/cpar -p 305 -d");


if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:s:p:l:dh") 
    din="" 
    dout=""
    dstar=""
    pSize=0;
    isDir=0;
    lFile="";
    for op, value in opts: 
        if op == "-i": 
            din = value 
        elif op == "-o": 
            dout = value
        elif op =="-s":
            dstar= value;
        elif op =="-p":
            pSize=int(value)
        elif op=="-d":
            isDir=1
        elif op=="-l":
            lFile=value;
        elif op == "-h": 
            usage() 
            din='./KLH1/data'
            dstar='./KLH1/star'
            dout='./KLH1/cpar'
            pSize=305
            isDir=1
            lFile='clist.txt'


    print(din,dout,dstar,pSize);
    if isDir ==1:
        comm=MPI.COMM_WORLD
        crank=comm.Get_rank();
        csize=comm.Get_size();

        if crank==0:
            if os.path.isdir(dout)==0:
                os.mkdir(dout);
            lFile=os.path.join(dout,lFile);
            lFile=open(lFile,'w');

        comm.barrier();


        fins=os.listdir(dstar);
        for i in range(crank,len(fins),csize):
            f=fins[i];
            if f[-4:]=='star' :
                fid=f[:-5]
                fin=fid+'.png';
                fin=os.path.join(din,fin);

                fstar=os.path.join(dstar,f);
                print('draw result:', fin,fstar,dout,lFile); 
                if os.path.exists(fin)==1 and  os.path.exists(fstar)==1:
                    fList=boxOne(fid,fin,fstar,pSize,dout);
                    lFile.writelines(fList);

        comm.barrier();
    else:
        boxOne(dstar[:-5], din,dstar,pSize,dout,lFile);

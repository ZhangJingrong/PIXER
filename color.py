import cv2
import sys, os,getopt 
from mpi4py  import MPI
from  EMAN2 import *
import numpy as np
def readStar(fin):
    f=open(fin,'r');
    l=f.readlines();
    count=0;
    for r in l:
        s=r.split();
        if len(s)>1:
            c=s[0][1];
            if c.isdigit()==1:
               break;
            count=count+1;
        else:
            count=count+1;

    l=l[count:];

    return l;



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

def norm(fin,n):
    em=EMData(fin);
    img=EMNumPy.em2numpy(em);
    arr2=normImg(img,n);
    return arr2;

def drawOri(im,cx,cy, sx,sy,count):
    startx=cx-sx/2;
    starty=cy-sx/2;
    endx=cx+sx/2;
    endy=cy+sx/2;

    cv2.rectangle(im, (startx, starty), (endx, endy), (255,0,0), 2)

def draw(im,cx,cy, sx,sy,r,g,b,shape,r1,g1,b1):
    startx=cx-sx/2;
    starty=cy-sx/2;
    endx=cx+sx/2;
    endy=cy+sx/2;

    w=3;

    if shape==0:
        cv2.rectangle(im, (startx, starty), (endx, endy), (b,g,r), w)
    elif shape==1:
       cv2.circle(im,(cx,cy), sx/2, (b,g,r), w)
    elif shape==2:
        cv2.rectangle(im, (startx, starty), (endx, endy), (b,g,r), w)
        cv2.circle(im,(cx,cy), 10, (b1,g1,r1), w)

def getCors(fstar):
    f=open(fstar,'r');
    plist=f.readlines();
    plist=plist[6:]
    return plist;


def drawOne(fin,fstar,pSize,fout,start,stop, step,shape,arrColor,color):
    plist=readStar(fstar);

    if(fin[-3:]=='png'):
        im = cv2.imread(fin);
    elif (fin[-3:]=='mrc'):
        im=norm(fin,2);
    
    sx=pSize;
    sy=pSize;

    sizex,sizey,sizez=im.shape;

    count=0;
    cx=0;
    cy=0;
    
    index=0;
    p=cp=0;
    for line in plist:
        if count>=start and count<stop:
             s= line.split()
             s= [item for item in filter(lambda x:x != '', s)];
             if len(s)>1:
                 x=float(s[0]);
                 x=int(x);
                 y=float(s[1]);
                 y=int(y);

                 if len(s)>=3:
                      p=float(s[2])
                 else:
                      p=1;

                 if len(s)==4:
                     cp=float(s[3]);
                 else:
                     cp=0;

                 if color==-1:
                     index=int((count-20)/step);
                     if index<0:
                         index=0;
                     elif index>9:
                         index=9;
                 else:
                     index=int(color);

                 r,g,b=arrColor[index]

                 if color==-1:
                     cdex=int((cp-0.5)*10);
                     if cdex<0:
                         cdex=0;
                 else:
                     cdex=int(color);
                 r1,g1,b1=arrColor[cdex];

                 draw(im,x,y,sx,sy,r,g,b,shape,r1,g1,b1);
        count=count+1;
    
    cv2.imwrite(fout,im);


def usage():
        print("mpiexec -n 6 -i /home/ict/dataset/objEmDb/experiment/empiar10005/data2 \
            -o /home/ict/dataset/objEmDb/experiment/empiar10005/rec \
            -s /home/ict/dataset/objEmDb/experiment/empiar10005/autopick-results-by-demo-type3-iter1-2 \
            -p 200 -d");

if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:s:p:z:q:c:n:f:e:dmh") 
    din="" 
    dout=""
    dstar=""
    pSize=0;
    isDir=0;
    shape=0;
    isMrc=-1;
    start=30;
    step=10;
    color=-1;
    stop=200000;

    for op, value in opts: 
        if op == "-i": 
            din = value 
        elif op == "-o": 
            dout = value
        elif op =="-s":
            dstar= value;
        elif op =="-p":
            pSize=int(value)
        elif op =="-z":
            start=float(value)
        elif op =="-e":
            stop=float(value)
        elif op =="-c":
            step=int(value)
        elif op =="-q":
            color=int(value)
        elif op =="-f":
            shape=int(value)
        elif op=="-d":
            isDir=1
        elif op=="-m":
            isMrc=1
        elif op=="-n":
            dnum=int(value);
        elif op == "-h": 
            usage() 
            din='/home/ict/pickyEye/empiar10075/relion/pickyEye/empiar10075/data/FoilHole_19046908_Data_19046157_19046158_20140520_0021_frames_SumCorr.png'
            dstar='/home/ict/pickyEye/empiar10075/relion/pickyEye/empiar10075/star/FoilHole_19046908_Data_19046157_19046158_20140520_0021_frames_SumCorr.star';
            dout='./test.png'
            start=50;
            step=10;
            shape=1;
            pSize=300;

    a=np.zeros([27,3]);
    index =0;
    
    a[0]=255,0,0;
    a[1]=0,0,255;
    a[2]=0,255,0;
    a[3]=255,0,255;
    a[4]=0,255,255;
    a[5]=255,255,0;
    a[6]=75,0,130;
    a[7]=0,100,0;
    a[8]=128,0,0;
    a[9]=128,128,0;
    a[10]=0,0,0;
    print(din,dout,dstar,pSize);
    if isDir ==1:
        comm=MPI.COMM_WORLD
        crank=comm.Get_rank();
        csize=comm.Get_size();

        if crank==0:
            if os.path.isdir(dout)==0:
                os.mkdir(dout);
        comm.barrier();


        fins=os.listdir(dstar);
        for i in range(crank,len(fins),csize):
            f=fins[i];
            if f[-4:]=='star' :
                fin=f[:-5]+'.png';
                fin=os.path.join(din,fin);

                if os.path.exists(fin)==0:
                    fin=f[:-5]+'.mrc';
                    fin=os.path.join(din,fin);
                if os.path.exists(fin)==1:
                    fstar=os.path.join(dstar,f);
                    fout=f[:-5]+'.png';
                    fout=os.path.join(dout,fout);
                    if os.path.exists(fstar)==1:
                        print(fstar, fin);
                        drawOne(fin,fstar,pSize,fout,start,stop, step, shape,a,color);
                else:
                    print('no file:', fin);

        comm.barrier();
    else:
        drawOne(din,dstar,pSize,dout,start, step, shape,a,color);

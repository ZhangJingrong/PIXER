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

def draw(den, im,cx,cy, sx,sy,count,total, c):
    startx=cx-sx/2;
    starty=cy-sx/2;
    endx=cx+sx/2;
    endy=cy+sx/2;

    if total ==0:
        p=1; 
    elif total<255 and  count<255:
        p=float(count)/total;
    elif count<255 and total>255:
        p=float(count)/255;
    else:
        p=1;
   

    r=c/4;
    g=(c-r*4)/2;
    b=c%2;
    r=r*255;
    g=g*255;
    b=b*255;

    w=2;
    
    cv2.rectangle(im, (startx, starty), (endx, endy), (b,g,r), w)
    
    #den[startx:endx, starty:endy]=den[startx:endx, starty:endy]+1;

    #return den;

def getCors(fstar):
    f=open(fstar,'r');
    plist=f.readlines();
    plist=plist[6:]
    return plist;


def drawOne(fin,fstar,pSize,fout,stop, color,dnum, isMrc):

    if isMrc<0:
        if(fin[-3:]=='png'):
            isMrc=0;
        elif (fin[-3:]=='mrc'):
            isMrc=1;

    plist=readStar(fstar);

    if(fin[-3:]=='png'):
        im = cv2.imread(fin);
    elif (fin[-3:]=='mrc'):
        im=norm(fin,2);
    
    sx=pSize;
    sy=pSize;

    sizex, sizey,sizez=im.shape;
    den=np.zeros([sizex,sizey]);

    count=1;
    cx=0;
    cy=0;

    for line in plist:
        if count>dnum:
            c=color+1;
        else:
            c=color;

        s= line.split()
        s= [item for item in filter(lambda x:x != '', s)];
        if len(s)>1:
             x=float(s[0]);
             x=int(x);
             y=float(s[1]);
             y=int(y);
             if len(s)==3:
                 p=float(s[2]);
             else:
                 p=1;

             if isMrc==1:
                 cx=x;
                 cy=y;
             elif isMrc==0:
                 cx=x;
                 cy=y;

             draw(den,im,cx,cy,sx,sy,count,len(plist), c);


             count=count+1;
    
    cv2.imwrite(fout,im);


def drawRec(fin, pSize, fout, color):
    isMrc=0;
    if(fin[-3:]=='png'):
        im = cv2.imread(fin);
    elif (fin[-3:]=='mrc'):
        im=norm(fin,2);
        isMrc=1;
    sx, sy,sz=im.shape;

    for x in range(pSize/2,sx,pSize):
        for y in range(pSize/2, sy, pSize):
            draw(im, x,y, pSize, pSize,0,0);
    cv2.imwrite(fout, im);


def usage():
        print("mpiexec -n 6 -i /home/ict/dataset/objEmDb/experiment/empiar10005/data2 \
            -o /home/ict/dataset/objEmDb/experiment/empiar10005/rec \
            -s /home/ict/dataset/objEmDb/experiment/empiar10005/autopick-results-by-demo-type3-iter1-2 \
            -p 200 -d");

if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:s:p:z:c:n:dmh") 
    din="" 
    dout=""
    dstar=""
    pSize=0;
    isDir=0;
    stop=0.6
    color=1;
    dnum=2000;
    isMrc=-1;
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
            stop=float(value)
        elif op =="-c":
            color=int(value)
        elif op=="-d":
            isDir=1
        elif op=="-m":
            isMrc=1
        elif op=="-n":
            dnum=int(value);
        elif op == "-h": 
            usage() 
            sys.exit()


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
                    #print('draw result:', fin,fstar,fout); 
                    if os.path.exists(fstar)==1:
                        drawOne(fin,fstar,pSize,fout,stop, color,dnum, isMrc);
                else:
                    print('no file:', fin);

        comm.barrier();
    else:
        drawOne(din,dstar,pSize,dout,stop, color,dnum, isMrc);
        #drawRec(dout,int(pSize*stop), dout, color);

from  EMAN2 import *
import sys,os,getopt
from matplotlib import pyplot as plt
import cv2
from skimage import io
from mpi4py  import MPI

import numpy as np

def norm(fin,fout,n):
    em=EMData(fin);
    img=EMNumPy.em2numpy(em);
    arr2=normImg(img,n);
    cv2.imwrite(fout, arr2);
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

    return arr2;

def getId(fname):
    lines=fname.split('/');
    fname=lines[-1];
    lines=fname.split('.');
    stop=-1*(len(lines[-1])+1)
    fid=fname[:stop];
    return fid;

def cutImg(fin,img,size,step,dout,flist):
    sizex,sizey=img.shape;
    f0=getId(fin);
    if sizex<size:
        size=sizex;
    oldx=-1000;
    oldy=-1000;
    start=0;
    for x in range(start,sizex,step):
        for y in range(start,sizey,step):
            startx=x;
            starty=y;
            if startx+size>sizex:
                startx=sizex-size;
            if starty+size>sizey:
                starty=sizey-size;
            if oldx !=startx or oldy!=starty:
                arr=img[startx:startx+size,starty:starty+size];
                fname=f0+'.'+str(startx)+'.'+str(starty)+'.'+str(size)+'.png';
                fout=os.path.join(dout,fname);
                cv2.imwrite(fout, arr);
                #name=dout.split('/');
                #name=name[-1];
                line=fname+'\n';
                flist.write(line);
                oldx=startx;
                oldy=starty;


def preProcess(fin,fout,dout,n,size,step,flist):
    img=norm(fin,fout,n);
    cutImg(fin,img,size,step,dout,flist)

def openList(flist,dout, comm):
    if os.path.isdir(dout)==0 and crank==0:
           os.mkdir(dout);
    comm.barrier();

    flist=os.path.join(dout,flist);
    flist=open(flist,'w');

    #dout=os.path.join(dout,'data');

    if os.path.isdir(dout)==0 and crank==0:
           os.mkdir(dout);
    comm.barrier();

    return dout,flist
def usage():
    print("python preProcess.py -i /home/ict/dataset/objEmDb/experiment/simu/pdb1f07/data \
            -o ./pdb1f07 -n 2 -s 512 -p 512 -l list.txt -d ");
    print("python preProcess.py -i /home/ict/dataset/objEmDb/experiment/simu/pdb1f07/data/pdb1f07-0.mrc \
            -o ./pdb1f07 -n 2 -s 512 -p 512 -l list.txt");


if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "hi:o:n:s:p:l:d")
    din=""
    dout=""
    n=3
    size=512
    step=512
    list_file="list.txt"
    isDir=1;
    
    for op, value in opts:
        if op == "-i":
            din = value
        elif op == "-o":
            dout = value
        elif op =="-n":
            n=float(value);
        elif op =="-s":
            size=int(value)
        elif op =="-p":
            step=int(value)
        elif op =="-l":
            list_file=value
        elif op == "-h":
            usage()
            sys.exit()
        elif op =="-d":
            isDir=1;mmcomm=MPI.COMM_WORLD
    comm=MPI.COMM_WORLD
    crank=comm.Get_rank();
    csize=comm.Get_size();

    print('mpi',crank,'getPara:', din,dout,n,size,step, list_file, isDir);


    if isDir==0:
        fout=getId(din)+'.png';
        fout=os.path.join(dout,fout);
        dout,flist=openList(list_file,dout,comm);
        fin=din;
        preProcess(fin,fout,dout,n,size,step,flist);
    else:
        fs=os.listdir(din);
        llist=list_file.split('.');
        list_file=list_file+'.'+str(crank);

        dout,flist=openList(list_file,dout,comm);


        for i in range(crank,len(fs),csize):
            f=fs[i];
            if f[-3:]=='mrc':
                fout=getId(f)+'.png';
                fout=os.path.join(dout,fout);
                fin=os.path.join(din,f);
                preProcess(fin,fout,dout,n,size,step,flist);

        flist.close();
        comm.barrier();


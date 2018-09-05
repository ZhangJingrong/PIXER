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
def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    star.write('_rlnParticleSelectZScore  #3\n');
    return star 


def minMax(fin):
    plist=readStar(fin);
    count=0;
    arr=np.ones(len(plist));

    for line in plist:
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

            arr[count]=cp;
            count=count+1;


    amax=np.max(arr);
    amin=np.min(arr);
    amean=np.mean(arr);
    astd=np.std(arr);
    minP=amean-astd;
    maxP=amean+astd;

    return maxP, minP,count;


def processOne(fin, fout,vstop):
    maxP, minP, count=minMax(fin);
    print(minP, maxP, count);
    fout=writeStarHead(fout);
    plist=readStar(fin);

    index=0;
    nCount=0;
    for line in plist:
        stopP=minP + vstop*(count-index)/count;

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
           
            if cp<stopP:
               nline=str(x)+' '+str(y)+' '+str(cp)+'\n';
               fout.write(nline);
               nCount=nCount+1;
    print('particle number: ', nCount);
    fout.close();

   

if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:p:d") 
    din='';
    dout='';
    vstop='';
    isDir=0;
    for op, value in opts: 
        if op == "-i": 
            din = value 
        elif op == "-o": 
            dout = value
        elif op =="-p":
            vstop=float(value)
        elif op =="-d":
            isDir=1;


    if isDir ==1:
        comm=MPI.COMM_WORLD
        crank=comm.Get_rank();
        csize=comm.Get_size();

        if crank==0:
            if os.path.isdir(dout)==0:
                os.mkdir(dout);
        comm.barrier();


        fins=os.listdir(din);
        for i in range(crank,len(fins),csize):
            f=fins[i];
            if f[-4:]=='star' :
                fin=os.path.join(din,f);
                fout=os.path.join(dout,f);
                processOne(fin, fout,vstop)
         comm.barrier();
    else:
        processOne(fin,fout);

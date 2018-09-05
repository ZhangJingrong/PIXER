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
def getKey(flist):
    fKey={};
    for line in flist:
        sline=line.split(' ');
        p=float(sline[1]);
        f=sline[0];

        s=f.split('/');
        fname=s[-1];

        fKey[fname]=p;
    return fKey;

def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    star.write('_rlnParticleSelectZScore  #3\n');
    star.write('_rlnParticleSelectCScore  #4\n');
    return star 
       
if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:s:dmh") 
    fin="" 
    dout=""
    dstar=""
    isDir=0;
    for op, value in opts: 
        if op == "-i": 
            fin = value 
        elif op == "-o": 
            dout = value
        elif op =="-s":
            dstar= value;
        elif op =="-d":
            isDir=1;
    print(fin ,dout,dstar);


    if isDir ==1:
        comm=MPI.COMM_WORLD
        crank=comm.Get_rank();
        csize=comm.Get_size();

        if crank==0:
            if os.path.isdir(dout)==0:
                os.mkdir(dout);
        comm.barrier();
            
        fp=open(fin,'r');
        flist=fp.readlines();
        fkey=getKey(flist);

        fins=os.listdir(dstar);
        for i in range(crank,len(fins),csize):
            f=fins[i];
            fid=f[:-5];
            fin=os.path.join(dstar,f);
            fout=os.path.join(dout,f);
            print(fin,fout);

            sp=writeStarHead(fout);
            slist=readStar(fin);
            for sline in slist:
                s=sline.split();
                x=float(s[0]);
                x=int(x);
                y=float(s[1]);
                y=int(y);
                pp=s[2];
                key=fid+'.'+str(x)+'.'+str(y)+'.png';
                if fkey.has_key(key):
                    p=fkey[key];
                    newline=str(x)+' '+str(y)+' '+pp+' '+str(p)+'\n';
                else:
                    newline=str(x)+' '+str(y)+' '+pp+' '+str(1)+'\n';
                sp.write(newline);
            sp.close();

        comm.barrier();

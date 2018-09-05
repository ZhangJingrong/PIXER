import os
import numpy as np
import sys, os,getopt 
from mpi4py  import MPI

def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    #star.write('_rlnScore  #3\n');
    return star 

def getFid(img):
    lines=img.split('/');
    fstar=lines[-1];
    ilines=fstar.split('.');
    sub=len(ilines[-1])+len(ilines[-2])+len(ilines[-3])+3;
    
    fid=fstar[:-1*sub];
    x=int(ilines[-2]);
    y=int(ilines[-3]);
    return fid,x,y;

    
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "f:o:i:s:d:h") 
    dout=""
    dpro=""
    vstop=0.6
    for op, value in opts: 
        if op == "-f": 
            fname = value 
        elif op == "-o": 
            dout = value
        elif op =="-i":
            dpro= value
        elif op =="-s":
            vstop=float(value);
        elif op=="-d":
            deploy_proto=value;
        elif op == "-h": 
            usage() 
            sys.exit()

    comm=MPI.COMM_WORLD
    crank=comm.Get_rank();
    csize=comm.Get_size();
    
    if crank==0:
        if os.path.isdir(dout)==0:
            os.mkdir(dout);
    comm.barrier();

    fpro='pro.'+str(crank)+'.txt';
    fpro=os.path.join(dpro,fpro);
    
    fp=open(fpro,'r');
    lines=fp.readlines();
    fold=0;
    for line in lines:
        sline=line.split();
        fname=sline[0];
        pro=float(sline[1]);
        fid, x,y =getFid(fname);

        if fid !=fold:
            if fold !=0:
                fstar.close();
            fstar=fid+'.star';
            fstar=os.path.join(dout,fstar);
            fstar=writeStarHead(fstar);
            fold=fid;

        if pro<vstop:
            wline=str(x)+' '+str(y)+'\n';
            fstar.write(wline);

    comm.barrier();

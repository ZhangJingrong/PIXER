# -*- coding: UTF-8 -*-
import os
os.environ['GLOG_minloglevel'] = '2'


import caffe
import numpy as np
import sys, os,getopt 
from mpi4py  import MPI

def writeStarHead(fstar):
    star=open(fstar,'w');
    star.write('data_\n\n');
    star.write('loop_\n');
    star.write('_rlnCoordinateX #1\n');
    star.write('_rlnCoordinateY #2\n');
    star.write('_rlnScore  #3\n');
    return star 

def getStarName(img):
    lines=img.split('/');
    fstar=lines[-1];
    ilines=fstar.split('.');
    sub=len(ilines[-1])+len(ilines[-2])+len(ilines[-3])+3;
    
    fstar=fstar[:-1*sub]+'.star';
    fstar=os.path.join(dout,fstar);
    print(fstar);
    
    return fstar;

def getFid(img):
    lines=img.split('/');
    fstar=lines[-1];
    ilines=fstar.split('.');
    sub=len(ilines[-1])+len(ilines[-2])+len(ilines[-3])+3;
    
    fid=fstar[:-1*sub];
    return fid;

def test(caffe_model, meanArr,din, dout,fpro,vstop, deploy_proto, fids):
    net = caffe.Net(deploy_proto, caffe_model, caffe.TEST)
    print(deploy_proto,caffe_model);

    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2,0,1))
    transformer.set_raw_scale('data', 255);
    transformer.set_channel_swap('data', (2,1,0)) ;
    transformer.set_mean('data', meanArr) 

    fpro=open(fpro,'w');

    for fname in fids:
        fstar=fname+'.star';
        fstar=os.path.join(dout,fstar);
        pfstar=writeStarHead(fstar);
        
        cm='ls '+din+fname+'* ';
        print(cm, crank);

        flist=os.popen(cm);
        for  line in  flist: 
            img=os.path.join(din, line);
            img=line[:-1];

            im = caffe.io.load_image(img)
            net.blobs['data'].data[...] = transformer.preprocess('data',im)
    
            out = net.forward()
            prob = net.blobs['prob'].data[0].flatten()
            order = prob.argsort()[-1]
            print(img,prob,order);
            line=img+' '+str(prob[0])+' \n';
            fpro.write(line);
            if prob[0]<vstop:
                ilines=img.split('.');
                x=ilines[-3];
                y=ilines[-2];
                p=str(prob);
                line=x+' '+y+' '+p+'\n';
                pfstar.write(line);
    fpro.close();

    
if __name__ == '__main__':
    opts, args = getopt.getopt(sys.argv[1:], "f:i:o:r:p:s:m:d:h") 
    fname="" 
    dout=""
    dref=""
    din=""
    dpro=""
    vstop=0.6
    caffe_model='/home/ict/git/classifier/netAlex2Way/snapshot/twoWay_iter_45000.caffemodel'
    deploy_proto ='/home/ict/git/classifier/netAlex2Way/deploy.prototxt';
    deploy_proto ='/home/ict/git/python/pickEye/cls.prototxt'; 
    for op, value in opts: 
        if op == "-f": 
            fname = value 
        elif op == "-i":
            din=value;
        elif op == "-o": 
            dout = value
        elif op =="-p":
            dpro= value
        elif op =="-r":
            dref=value;
        elif op =="-s":
            vstop=float(value);
        elif op=="-m":
            caffe_model=value;
        elif op=="-d":
            deploy_proto=value;
        elif op == "-h": 
            usage() 
            sys.exit()

    meanArr=np.zeros(3);
    meanArr[0]=meanArr[1]=meanArr[2]=123.834;
    
    comm=MPI.COMM_WORLD
    crank=comm.Get_rank();
    csize=comm.Get_size();
    
    if crank==0:
        if os.path.isdir(dout)==0:
            os.mkdir(dout);
        if os.path.isdir(dpro)==0:
            os.mkdir(dpro);
    comm.barrier();

    fins=os.listdir(dref);
    fids=[];
    for i in range(crank,len(fins),csize):
        fin=fins[i];
        lines=fin.split('/');
        fid=lines[-1];
        fid=fid[:-4]
        fids.append(fid);
   
    fpro='pro.'+str(crank)+'.txt';
    fpro=os.path.join(dpro,fpro);

    test(caffe_model,meanArr,din,dout,fpro,vstop,deploy_proto,fids);
    
    comm.barrier();

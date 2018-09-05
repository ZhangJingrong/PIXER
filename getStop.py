import sys, os,getopt 
from mpi4py  import MPI
def readStar(fin):
    f=open(fin,'r');
    l=f.readlines();

    count=0;
    for r in l:
        if len(r)>2:
            c=r[1];
            count=count+1;
            if c.isdigit()==1:
                break;
    print(count);
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

def getScore(l):
    pList=[];
    for line in l:
        s=line.split();
        p=float(s[2]);
        print(p);
        pList.append(p);

    return pList;


def stopOne(fin, fout, nStop, percent):
    clist=readStar(fin);
    length=len(clist);
    nPer=int(length*percent);
    if nPer>nStop:
        nPer=nStop;
    nlist=clist[:nPer];
    fp=writeStarHead(fout)
    fp.writelines(nlist);
    fp.close();



def usage():
    print('python getStop.py \
            -i /home/ict/pickyEye/empiar10075/relion/pickyEye/empiar10075/star/FoilHole_19046908_Data_19046157_19046158_20140520_0021_frames_SumCorr.star\
            -o ./test.star -s 200 -p 0.6 ');


if __name__=="__main__":
    opts, args = getopt.getopt(sys.argv[1:], "i:o:s:p:dh") 
    din="" 
    dout=""
    isDir=0;
    nStop=200;
    percent=0.8;
    for op, value in opts:
        if op == "-i":
            din = value
        elif op == "-o":
            dout = value
        elif op == "-s":
            nStop = int(value)
        elif op == "-p":
            percent=float(value)
        elif op=="-d":
            isDir=1
        elif op == "-h":
            usage()
            din='/home/ict/pickyEye/empiar10075/relion/pickyEye/empiar10075/star/FoilHole_19046908_Data_19046157_19046158_20140520_0021_frames_SumCorr.star';
            dout='./test.star'
            nStop=200
            percent=0.6

    print(din ,dout, nStop, percent);
    comm=MPI.COMM_WORLD
    crank=comm.Get_rank();
    csize=comm.Get_size();



    if isDir ==1:
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

                print(fin,fout); 
                stopOne(fin, fout, nStop, percent)

        comm.barrier();
    else:
        stopOne(din, dout, nStop, percent)


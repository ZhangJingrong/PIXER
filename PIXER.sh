mrcArr=(testDataSet)
dinArr=('./testDataSet/mrc2' )
sizeArr=(128)
numArr=(250)
gpuID=0
mpiNum=6

pre=/home/jingrong/software/EM/Pixer/PIXER
num=6 

start=0;
end=1;
PRECLEAN=0
PRE=1
RUN_CAFFE=1
POST=1
CONTOUR=1
STOP=1
CLASS=1
DRAW=1
VAL=1
CLEAN=0
ISCROSS=0

process=$PRECLEAN$PRE$RUN_CAFFE$POST$CONTOUR$STOP$CLASS$DRAW$VAL$CLEAN$ISCROSS
echo $process

for ((i=start;i<end;i++))
do
   mrc_din=${dinArr[$i]}
   mrcId=${mrcArr[$i]}
   size=${sizeArr[$i]}
   pnum=${numArr[$i]}
   dout=${pre}/$mrcId
   cm="bash ${pre}/run.sh -f $mrcId -i $mrc_din -s $size -o $dout  -n $pnum -g $gpuID -r $mpiNum -d -b $process "
   echo $cm
   $cm
done



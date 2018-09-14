#/bin/bash
PRE=1
RUN_CAFFE=1
POST=1
CLEAN=0
BIN=0
CONTOUR=0
DRAW=0
EVAL=0
STOP=0
NRANK=1
PRECLEAN=0

mrcId=pdb1f1h
mrc_din=/home/ict/dataset/objEmDb/experiment/simu/${mrcId}/data/
psize=100
pnum=2500
normlize=2

cutImgSize=512
cutImgStep=256

listFile=list.txt
logFile=log.txt

isDir='-d'
isPrint='-z'
isKeep=0

stopValue=0.1
IF_TIME=1
gpuId=0
vstop=0.9
pStop=0.6
pickyEye=/home/jingrong/software/EM/Pixer/PIXER
caffe_din=/home/jingrong/software/caffe
caffe_din=/home/jingrong/software/deepLabV2
gCan=0.1
nIter=500
nSep=40
oLap=0.7



while getopts "i:f:s:n:o:g:r:v:m:l:p:q:dtb:v:w:" arg
do
    case $arg in
        f)
            echo "Id: $OPTARG"
            mrcId=$OPTARG;
            ;;
        i)
            echo "Input Directory: $OPTARG"
            mrc_din=$OPTARG;
            ;;
        s)
            echo "Size of particle: $optarg $OPTARG "
            psize=$OPTARG;
            ;;
        n)
            echo "Num of particle: $OPTARG"
            pnum=$OPTARG
            ;;
        o)
            echo "Output directory:  $OPTARG "
            result_dout=$OPTARG
            ;;
        g)
            echo "Gpu ID:  $OPTARG "
            gpuId=$OPTARG
            ;;
        r)
            echo "Num of MPI Process: $OPTARG"
            NRANK=$OPTARG
            ;;
        m)
            echo "Candidate Step: $OPTARG"
            gCan=$OPTARG
            ;;
        l)
            echo "Local Maximum Iteration: $OPTARG"
            nIter=$OPTARG
            ;;
        p)
            echo "Dilate Iterations: $OPTARG"
            nSep=$OPTARG
            ;;
        q)
            echo "Remove Overlap Particles: $OPTARG"
            oLap=$OPTARG
            ;;
        w)
            echo "The Percent of Local Result Remained: $OPTARG"
            pStop=$OPTARG
            ;;
        d)
            echo "Running For Directory"
            isDir='-d'
            ;;
        t)
            echo "Record Time"
            IF_TIME=1
            ;;
        v)
            echo "Stop Value"
            vstop=$OPTARG
            ;;

        b)
            com=$OPTARG
            PRECLEAN=${com:0:1}
            PRE=${com:1:1}
            RUN_CAFFE=${com:2:1}
            POST=${com:3:1}
            CONTOUR=${com:4:1}
            STOP=${com:5:1}
            CLASS=${com:6:1}
            DRAW=${com:7:1}
            EVAL=${com:8:1}
            CLEAN=${com:9:1}
            echo "Choose the Job preclean: $PRECLEAN preProcess: $PRE segNet: $RUN_CAFFE postProcess: $POST contour: $CONTOUR truncation: $STOP classNet: $CLASS draw: $DRAW clean: $CLEAN "
            ;;
        ?)
            echo "ERROR:unkonw argument"
            exit 1
        esac
done


TIME=
if [ $IF_TIME -eq 1 ]; then
    TIME="/usr/bin/time -p"
fi

if [ $PRECLEAN -eq 1 ]; then
    if [ -d ${result_dout}/data ]; then
        rm -r ${result_dout}/heat*;
        rm -r ${result_dout}/rec*;
        rm -r ${result_dout}/star*;
        rm -r ${result_dout}/mat*;
        rm -r ${result_dout}/draw*;
        #rm -rf ${result_dout}/data/list*;
        #rm -r ${result_dout}
       echo remove dir ${result_dout}
    fi
fi

if [ ! -n $result_dout ]; then
    result_dout=${pickyEye}/${mrcId}
fi

if [ ! -d $result_dout  ]; then
        mkdir -p $result_dout;
    fi


if [ $PRE -eq 1 ]; then
       cm="$TIME mpiexec -n $NRANK python ${pickyEye}/mpiPre.py -i ${mrc_din} -o ${result_dout}/data/ \
           -n ${normlize} -s ${cutImgSize} -p ${cutImgStep} -l ${listFile} ${isDir} ${isPrint}"
       echo $cm& $cm
       echo $cm >${result_dout}/$logFile
       cat ${result_dout}/data/${listFile}.? >${result_dout}/${listFile}
   fi


if [ $RUN_CAFFE -eq 1 ]; then
       model="$pickyEye/whole512-new_train_iter_9800.caffemodel"
       export GLOG_minloglevel=1
       cm="$TIME bash ${pickyEye}/runNetwork.sh ${result_dout}/data/ ${result_dout}/${listFile} ${result_dout}/mat/ ${gpuId} ${model} ${caffe_din} ${cutImgSize} ${pickyEye}"
       echo $cm 
       $cm

       echo $cm >>${result_dout}/$logFile
       export GLOG_minloglevel=3

       #if [ $isKeep -eq 0 ]; then
           #rm  ${result_dout}/data/*png
       #fi

fi


if [ $POST -eq 1 ]; then
    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/mpiPost.py -i ${result_dout}/mat/  -o ${result_dout}/heat/ -b ${result_dout}/bin/ \
        -r ${result_dout}/data/ -l  ${result_dout}/data/${listFile} \
        -p ${psize} -n ${pnum} -s ${result_dout}/star/  ${isDir} ${isPrint}"
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    if [ $isKeep -eq 0 ]; then
        rm -r ${result_dout}/mat
    fi

fi

if [ $CONTOUR -eq 1 ]; then
    cm="$TIME mpiexec -n 1 python ${pickyEye}/gpuNonMax.py  -i ${result_dout}/bin/  -o ${result_dout}/rec/  -r  ${result_dout}/heat/ \
        -s ${result_dout}/star/ -p $psize -g $gCan -t $nIter -y $oLap -x $nSep ${isDir} ${isPrint}"

    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    #if[ $isKeep -eq 0 ]; then
    #    rm -r ${result_dout}/heat
    #fi

fi

if [ $STOP -eq 1 ]; then
    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/getStop.py  -i ${result_dout}/star/  -o ${result_dout}/star/ ${isDir} -p $pStop -s $pnum ${isPrint}"
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    if [ $isKeep -eq 0 ]; then
        rm -r ${result_dout}/rec
    fi

fi

if [ $CLASS -eq 1 ]; then
    model="${pickyEye}/twoWay_iter_45000.caffemodel"
    deploy="${pickyEye}/deploy.prototxt"

    cm="$TIME mpiexec -n 1 python ${pickyEye}/genPar.py  -i ${result_dout}/data/ -s ${result_dout}/star/  -o ${result_dout}/cpar/ \
        -l ${listFile} -p $psize ${isDir} ${isPrint}"
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/testCls.py -i ${result_dout}/cpar/ -f ${result_dout}/cpar/${listFile} \
        -r  ${result_dout}/heat/ -o ${result_dout}/cstar/ -p ${result_dout}/cpro/ -s 0.9 -m $model -d $deploy ${isPrint}"
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    #cm="$TIME mpiexec -n $NRANK python ${pickyEye}/erasePar.py -i ${result_dout}/cstar/ -o ${result_dout}/cstarClean/ -s $vstop"
    #echo $cm & $cm
    #echo $cm >>${result_dout}/$logFile

    if [ $isKeep -eq 0 ]; then
        rm -r ${result_dout}/cpar
    fi

fi

if [ $DRAW -eq 1 ]; then
    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/drawRec.py  -i ${result_dout}/data/  -o ${result_dout}/drawLocal/  -s ${result_dout}/star/ \
        -p $psize -y $stopValue -c 1 ${isDir} -n $pnum $isPrint "
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile

    #cm="$TIME mpiexec -n $NRANK python ${pickyEye}/drawRec.py  -i ${result_dout}/data/  -o ${result_dout}/cdraw/  -s ${result_dout}/cstarClean/ \
    #    -p $psize -y $stopValue  -c 1 ${isDir} -n $pnum"
    #echo $cm & $cm
    #echo $cm >>${result_dout}/$logFile
fi

if [ $EVAL -eq 1 ]; then
    cat  ${result_dout}/cpro/pro.?.txt >${result_dout}/cpro/pro.txt
    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/cscore.py -i ${result_dout}/cpro/pro.txt \
        -o ${result_dout}/cScore/ -s ${result_dout}/star/ $isDir $isPrint"
    echo $cm & $cm
    echo $cm >>${result_dout}/$logFile
    
    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/cleanPar.py  -i  ${result_dout}/cScore/ -o ${result_dout}/cScore2/ -p 0.8 $isDir $isPrint"
    echo $cm & $cm

    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/color.py  -i  ${result_dout}/data/ \
        -o ${result_dout}/colorDraw/  -s ${result_dout}/cScore/ \
        -p $psize -y 20 -c 10 -q -1 -f 1 $isDir $isPrint"
    echo $cm & $cm

    cm="$TIME mpiexec -n $NRANK python ${pickyEye}/color.py  -i  ${result_dout}/data/ \
        -o ${result_dout}/colorDraw2/  -s ${result_dout}/cScore2/ \
        -p $psize -y 0 -c 10 -q -1 -f 1 $isDir $isPrint"
    #echo $cm & $cm

fi

if [ $CLEAN -eq 1 ]; then
    rm -r  ${result_dout}/data
    rm -r  ${result_dout}/mat/
    rm -r  ${result_dout}/heat/
    rm -r  ${result_dout}/bin/
    rm -r  ${result_dout}/rec/
    rm -r  ${result_dout}/cpar/
    rm -r  ${result_dout}/cpro/
    rm -r  ${result_dout}/cstar/
    rm ${result_dout}/${listFile}
fi


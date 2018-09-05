import numpy as np
f=open('record.txt');
lines=f.readlines();
results=[];
sum=0;
step=int(100/6.168);

for line in lines:
    s=line.split();
    print(len(s));

    for i in s:
        res=int(i);
        res=int(res/6.168);
        results.append(res);

print(results);

length=len(results);
fp=np.zeros(7);
k=0;

for i in range(0,7):
    sum=0;
    j=0;
    while j<100 and k<length:
        sum=sum+results[k];
        k=k+1;
        j=j+1;
    fp[i]=sum;
    i=i+1;
print(fp, np.sum(fp));

f.close();

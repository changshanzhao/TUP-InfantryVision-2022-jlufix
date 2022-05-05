#!/bin/bash 

source /opt/intel/openvino_2021.4.752/bin/setupvars.sh
sec=1 
cnt=0 
name=TUP-InfantryVision-2022
progarm_name=Infantry_Vision
cd /home/dishierweidu/Desktop/$name/build/
#make clean && 
make -j6 
while [ 1 ] 
do 
    count=`ps -ef | grep $name | grep -v "grep" | wc -l`
    echo "Thread count: $count" 
    echo "Expection count: $cnt" 
    if [ $count -ge 1 ]; then 
        echo "The $name is still alive!" 
        sleep $sec 
    else  
        echo "Starting $name..." 
        gnome-terminal -- bash -c "cd /home/tup/Desktop/$name/build/;
        ./$name;exec bash;" 
        echo "$name has started!"   
        sleep $sec 
        ((cnt=cnt+1)) 
        if [ $cnt -gt 9 ]; then 
            #reboot 
        fi 
    fi 
done

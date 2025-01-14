#!/bin/bash 

source /opt/intel/openvino_2023.0.1/setupvars.sh
sec=2 
cnt=0 
name=TUP-InfantryVision-2022-jlufix
program_name=Infantry_Vision
cd /home/bubing2/桌面/$name/build/
#make clean && 
make -j6 
while [ 1 ] 
do 
    count=`ps -ef | grep $program_name | grep -v "grep" | wc -l`
    echo "Thread count: $count" 
    echo "Expection count: $cnt" 
    if [ $count -ge 1 ]; then 
        echo "The $name is still alive!" 
        sleep $sec 
    else  
        echo "Starting $name..." 
        gnome-terminal -- bash -c "cd /home/bubing2/桌面/$name/build/;
        ./$program_name;exec bash;" 
        echo "$name has started!"   
        ((cnt=cnt+1)) 
        sleep $sec 
        if [ $cnt -gt 9 ]; then 
            echo "Reboot!" 
            #reboot 
        fi 
    fi 
done

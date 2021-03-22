#!/bin/bash
echo "Hello from Mamatus team !"
cd /home/pi/ocoda
source ./bin/activate

python animator.py -p "test" -g "False"



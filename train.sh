#!/usr/bin/env bash

nohup caffeinate -i nice -19 python ./src/train.py < /dev/null > ./data/train.dat 2>./data/train.log &

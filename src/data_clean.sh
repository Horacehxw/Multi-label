#!/bin/bash
# find all the "*.txt" under this dir and pass them one by one to the data_claen
# program. all output and stderr are in .log file.
find -name "*.txt" | xargs -n 1 python data_clean.py >> data_cleam.log 2>&1

import sys
from data_util import DataPoint
from data_util import read_a_point
import sys


filename = ""

if len(sys.argv) >= 2:
    filename = str(sys.argv[1])
    print("\n\n--------------------------\nProcessing the file: {}\n".format(filename))
else:
    print("usage: python clean_data.py filename\n")

print("pass one: count the features \n ... ... ...")
total_point, num_features, num_labels = 0,0,0
with open(filename, "r") as file: #too much data to read all at the same time
    first_line = file.readline().split()
    first_line = [int(x) for x in first_line]
    total_point, num_features, num_labels = first_line[0], first_line[1], first_line[2]
    d_points = []
    feature_counter = {}
    for line in file:
        read_a_point(line, counter=feature_counter) # calculating the features
    #d_points.append(d_point)
print("pass one comlete!\n")

prominent_feature = set()
for feature in feature_counter:
    if feature_counter[feature] >= total_point / 2:
        prominent_feature.add(feature)
print ("prominent_features are:", prominent_feature)

print("pass two: read the data points with prominent features\n ... ... ...")
d_points = []
with open(filename, "r") as input_file:
    first_line = input_file.readline()
    for line in input_file:
        d_points.append(read_a_point(line, prominent=prominent_feature))
print("pass two complete!\n")

print("storing the cleaned data\n ... ... ...")
with open(filename+".output", "w") as output_file:
    first_line = str(total_point)+" " \
        +str(len(prominent_feature))+" "+str(num_labels)+"\n"
    output_file.write(first_line)
    for d_point in d_points:
        output_file.write(str(d_point))
print("file stord in " + filename + ".output\n----------------------------------\n")

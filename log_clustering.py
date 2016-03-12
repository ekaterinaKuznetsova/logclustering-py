'''
This program is used to cluster logs based on their similarity. 
The similarity between two logs is measured based the editing distance between. 
The basic unit of a log is not a character but a token. 

[1] Wikibooks: Algorithm Implementation/Strings/Levenshtein distance - Python Code:
https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

[2] Fast implementation of the edit distance(Levenshtein distance)
https://github.com/aflc/editdistance

The method 'levenshteinVec(source, target)' implemented here is very slower than the package 'editdistance',
Since the source code of 'editdistance' is written in C++
However, if we would like to modify the levenshtein algorithm by adding weights to different token classes
'levenshteinVec(source, target)' is easier to change the code

TODO: pre-partition. output sequence. 
'''

import sys
import numpy as np
import timeit
import re
import time
import editdistance



logfile = "/home/cliu/Documents/SC-1/message"
outfile = "/home/cliu/Documents/SC-1/output"
console = "/home/cliu/Documents/SC-1/console"

# delimiters for dividing a log into tokens
delimiter = r'[ ,:()\[\]=|/\\{}\'\"]+' # ,:()[]=|\/{}'"          no <>

# two logs with editing distance less than distance_threshold are considered to be similar
distance_threshold = 0.4

# how many chars are ignored from the beginning of a log (because of the time-stamp, server-name, etc.)
ignored_chars = 21




def levenshteinVec(source, target):
    '''
    Dynamic Programming algorithm, with the added optimization that 
    Only the last two rows of the dynamic programming matrix are needed for the computation
    Vectorized version using NumPy. 
    '''
    if len(source) < len(target):
        return levenshteinVec(target, source)

    # So now we have len(source) >= len(target).
    if len(target) == 0:
        return len(source)

    # We call tuple() to force strings to be used as sequences
    # ('c', 'a', 't', 's') - numpy uses them as values by default.
    source = np.array(tuple(source))
    target = np.array(tuple(target))

    # We use a dynamic programming algorithm, but with the
    # added optimization that we only need the last two rows
    # of the matrix.
    previous_row = np.arange(target.size + 1)
    for s in source:
        # Insertion (target grows longer than source):
        current_row = previous_row + 1

        # Substitution or matching:
        # Target and source items are aligned, and either
        # are different (cost of 1), or are the same (cost of 0).
        current_row[1:] = np.minimum(
                current_row[1:],
                np.add(previous_row[:-1], target != s))

        # Deletion (target grows shorter than source):
        current_row[1:] = np.minimum(
                current_row[1:],
                current_row[0:-1] + 1)

        previous_row = current_row

    return (float(previous_row[-1]) / float(max(len(source), len(target))))



def isTimeStamp(input_str):
    '''
    Check whether input str is with time format like: Feb 11 05:22:51
    '''
    try:
        time.strptime(input_str, '%b %d %H:%M:%S ')
        return True
    except ValueError:
        return False



def minDistance(added_line, cluster_dict):
    '''
    Calculate the minimal distance between the log and all the clusters
    Return the minimal distance and its index
    '''
    distance = []
    added_line_tokens = replaceNumByWildcard(re.split(delimiter, added_line[ignored_chars:]))
    for cluster_num in cluster_dict:
        cluster = cluster_dict[cluster_num]
        cluster_line_tokens = replaceNumByWildcard(re.split(delimiter, cluster[0][ignored_chars:]))
        dis_ratio = (float(editdistance.eval(cluster_line_tokens, added_line_tokens)) / float(max(len(added_line_tokens), len(cluster_line_tokens))))
        distance.append(dis_ratio)
    #print distance
    min_index = np.argmin(distance)
    min_dis = distance[min_index]

    return min_dis, min_index



def replaceNumByWildcard(tokens):
    for i in range(0, len(tokens)):
        if tokens[i].isdigit() or re.search(r'0x[\da-fA-F]', tokens[i]) is not None:
            tokens[i] = '*'   
    return tokens

     
     

def partitionByCommand():
    command_cluster = {}
    pattern = re.compile(r'(\w+)([\[:])(.*)') 
    
    with open(logfile) as f:
        with open(outfile, 'w') as o:
            added_line = f.readline()
            while not isTimeStamp(added_line[:16]):
                added_line = f.readline()
            for line in f:
                if not isTimeStamp(line[:16]):
                    added_line = added_line.rstrip() + ' | ' + line
                    continue
                else:        
                    # Do something for each log
                    #o.write(added_line)                   
                    #o.write((re.match(pattern, added_line[21:])).group(1) + '\n')
                    command = (re.match(pattern, added_line[21:])).group(1)
                    if not command_cluster.has_key(command):
                        command_cluster[command] = [added_line]
                    else:
                        command_cluster[command].append(added_line)
        
                    added_line = line
                            
            # Add the last line        
            # Do something for the last log
            #o.write(added_line)
            #o.write((re.match(pattern, added_line[21:])).group(1) + '\n')
            command = (re.match(pattern, added_line[21:])).group(1)
            if not command_cluster.has_key(command):
                command_cluster[command] = [added_line]
            else:
                command_cluster[command].append(added_line)
    
    
    with open(console, 'w') as c:
        for i in command_cluster:
            c.write(str(i) + '\n')  
            for item in command_cluster[i]:
                c.write(item)  
                
    return command_cluster
     
     
def logClusteringWithPrePartition():
    command_cluster = partitionByCommand()
    
    with open(console, 'w') as c:
        for i in command_cluster:
            c.write(str(i) + '\n')  
            for item in command_cluster[i]:
                c.write(item)    
    
    
     
     
     
def logClustering():

    cluster_dict = {}
     
    with open(logfile) as f:
        with open(outfile, 'w') as o:
            added_line = f.readline()
            while not isTimeStamp(added_line[:16]):
                added_line = f.readline()
            for line in f:
                if not isTimeStamp(line[:16]):
                    added_line = added_line.rstrip() + ' | ' + line
                    continue
                else:        
                    # Do something for each log
                    #o.write(added_line)                   
                    if not cluster_dict:
                        cluster_dict[0] = [added_line]
                    else:
                        #cluster_dict[len(cluster_dict)] = [added_line]
                        min_dis, min_index = minDistance(added_line, cluster_dict)
                        if min_dis < distance_threshold:
                            cluster_dict[min_index].append(added_line)
                        else:
                            cluster_dict[len(cluster_dict)] = [added_line]
        
                    added_line = line
                            
            # Add the last line        
            # Do something for the last log
            #o.write(added_line)
            min_dis, min_index = minDistance(added_line, cluster_dict)
            if min_dis < distance_threshold:
                cluster_dict[min_index].append(added_line)
            else:
                cluster_dict[len(cluster_dict)] = [added_line]
    
    
    #sys.stdout = open(console, 'w')
    with open(console, 'w') as c:
        for i in cluster_dict:
            c.write(str(i) + '\n')  
            for item in cluster_dict[i]:
                c.write(item)    
     
     
     
     
     
            
            
            
            
def main():
 
    start_time = time.time()
    #logClustering()
    partitionByCommand()
    stop_time = time.time()
    
    print "--- %s seconds ---" % (stop_time - start_time)
    
    
    # ------------------------------ For debugging ------------------------------ #
    #list1 = ['a', 'b', 'c', 'd']
    #list2 = ['a', 'b', 'd', 'c', 'e']   
    #print levenshteinVec(list1, list2) 
    #str1 = "SC-1 ecimswm: ActivateUpgradePackage::doAct`~ion: com<>pleted 2 of 2 procedures (100 percent)"   
    #print replaceNumByWildcard(re.split(delimiter, str1))         
    #string1 = 'adqe!f-'
    #string2 = 'adqef-'
    #print levenshteinVec(string1, string2)
    #print isTimeStamp("Feb 17 04:16:54 a"[:16])
    #with open(logfile) as f:
    #    print str(f.readline()).endswith('\n')
    #print bool({})
    
    #print re.search(r'0x[\da-fA-F]', "0x0e") is not None
    
    #print editdistance.eval(list1, list2)
    
    # ------------------------------ For debugging ------------------------------ #
    
    print "Stop..."
    





if __name__ == "__main__":
    #sys.stdout = open(console, 'w')
    main()
    #print(timeit.timeit("main()", number=100 ,setup="from __main__ import main"))          
            
            




























'''
This program is used to cluster logs based on their similarity. 

The similarity between two logs is measured based the editing distance between. 

The basic unit of a log is not a character but a token. 

Wikibooks: Algorithm Implementation/Strings/Levenshtein distance - Python Code:
https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python
'''

import numpy as np
import timeit

logfile = "/home/cliu/Documents/SC-1/messages"
outfile = "/home/cliu/Documents/SC-1/output"
    


def levenshtein(s1, s2):
    '''
    Dynamic Programming algorithm, with the added optimization that 
    Only the last two rows of the dynamic programming matrix are needed for the computation
    '''
    if len(s1) < len(s2):
        return levenshtein(s2, s1)

    # len(s1) >= len(s2)
    if len(s2) == 0:
        return len(s1)

    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1 # j+1 instead of j since previous_row and current_row are one character longer
            deletions = current_row[j] + 1       # than s2
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    
    return (float(previous_row[-1]) / float(max(len(s1), len(s2))))



def levenshteinVec(source, target):
    '''
    Vectorized version of the levenshtein(s1, s2):, using NumPy. 
    About 40% faster based on some test case.
    '''
    if len(source) < len(target):
        return levenshtein(target, source)

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






            
def main():
    with open(logfile) as f:
        with open(outfile, 'w') as o:
            curr_line = f.readline()[16:]
            o.write(str(levenshteinVec(curr_line, curr_line)) + '  ' + curr_line)
            for line in f:
                line = line[16:]
                edit_dis = levenshteinVec(curr_line, line)
                if edit_dis < 0.1:
                    o.write(str(edit_dis) + '  ' + line)
                
                

if __name__ == "__main__":
    main()
    #print(timeit.timeit("main()", number=10 ,setup="from __main__ import main"))          
            
            





























#13
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

#10
def minimumEditDistance(s1,s2):
    if len(s1) > len(s2):
        s1,s2 = s2,s1
    distances = range(len(s1) + 1)
    for index2,char2 in enumerate(s2):
        newDistances = [index2+1]
        for index1,char1 in enumerate(s1):
            if char1 == char2:
                newDistances.append(distances[index1])
            else:
                newDistances.append(1 + min((distances[index1],
                                             distances[index1+1],
                                             newDistances[-1])))
        distances = newDistances
    #return distances[-1]
    return (float(distances[-1]) / float(max(len(s1), len(s2))))

#16
def levenshteinDistance(str1, str2):
    m = len(str1)
    n = len(str2)
    lensum = float(m + n)
    d = []
    for i in range(m+1):
        d.append([i])
    del d[0][0]
    for j in range(n+1):
        d[0].append(j)
    for j in range(1,n+1):
        for i in range(1,m+1):
            if str1[i-1] == str2[j-1]:
                d[i].insert(j,d[i-1][j-1])
            else:
                minimum = min(d[i-1][j]+1, d[i][j-1]+1, d[i-1][j-1]+2)
                d[i].insert(j, minimum)
    ldist = d[-1][-1]
    ratio = (lensum - ldist)/lensum
    #return {'distance':ldist, 'ratio':ratio}
    return (float(ldist) / float(max(len(str1), len(str1))))


    # if command not in command_cluster:
    #     command_cluster[command] = [added_line]
    # else:
    #     command_cluster[command].append(added_line)
    #     
    # ---------------------------- For debugging ---------------------------- #
    # list1 = ['a', 'b', 'c', 'd']
    # list2 = ['a', 'b', 'd', 'c', 'e']
    # print extractor.levenshtein_numpy(list1, list2)
    # str1 = ("SC-1 ecimswm: ActivateUpgradePackage::doAct`~ion: "
            # "com<>pleted 2 of 2 procedures (100 percent)")
    # print extractor.to_wildcard(re.split(extractor.delimiter, str1))
    # string1 = 'adqe!f-'
    # string2 = 'adqef-'
    # print extractor.levenshtein_numpy(string1, string2)
    # print extractor.is_time("Feb 17 04:16:54 a"[:16])
    # with open(logfile) as f:
    #     print str(f.readline()).endswith('\n')
    # print bool({})
    #
    # print editdistance.eval(list1, list2)

    # print extractor.is_ipv4("127.1.1.1")
    # # print extractor.is_ipv4("2001:1b70:82a0:46c::")
    # print extractor.is_ipv6("fe80::1:a")
    # print extractor.is_ipv6("2001:1b70:82a0:46c::0")
    # print extractor.is_ipv6("0000:00:")
    # print extractor.is_ipv6("fe80::213:5eff:feea:294c")
    # print extractor.is_ipv6("fe80::200:ff:feff:1")
    #
    #
    # print extractor.is_number('ccb')
    #
    # print extractor.is_time("Feb 17 04:16:54.16154")
    # print extractor.is_time("Feb 7 04:16:54")
    # print extractor.is_time("Feb 17 4:16:54")
    # print extractor.is_time("2016-02-16T18:23:34")
    # print extractor.is_time("04:16:54")
    # print extractor.is_time("0000:00:")

    # print extractor.is_pci_address("a0000:ff:0a.1:")

    # print extractor.to_wildcard(['a', '0xa', '0', '0.0', '0xo', 'b',
                                #  '125.6.3.2', '125.6.3.256', '0000:35:25.1:'])

    # print 5*1.0/13
    # print extractor.is_time("Feb 17 04:16:54 p")

    # pattern = re.compile(r'([\w\-\_\./]+)([\[:])(.*)')
    #
    # print re.match(pattern, astr1[21:]).group(1)

    # print ''.join(re.split(r'([ ,:()\[\]=|/\\{}\'\"<>]+)', str1))

    # ---------------------------- For debugging ---------------------------- #

"""
This program is used to cluster logs based on their similarity.
The similarity between two logs is measured based the editing distance between.
The basic unit of a log is not a character but a token.

[1] Wikibooks: Algorithm Implementation/Strings/Levenshtein distance - Python:
https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

[2] Fast implementation of the edit distance (Levenshtein distance) - C++:
https://github.com/aflc/editdistance

[3] The dateutil module provides powerful extensions to the standard datetime
module, available in Python.
https://github.com/dateutil/dateutil

The method 'levenshteinNumPy(source, target)' implemented here is very slower
than the package 'editdistance', since the source code of 'editdistance' is
written in C++. However, if we would like to modify the levenshtein algorithm
by adding weights to different token classes 'levenshteinNumPy(source, target)'
is easier to change the code
"""


import editdistance
from dateutil.parser import parse as timeparser
# import mlpy
# from numbers import Number
import numpy as np
import re
# import sys
import socket
import time


class LogTemplateExtractor(object):
    """
    A log template extractor
    """
    def __init__(self, logfile):
        self.logfile = logfile
        self.outfile = "/home/cliu/Documents/SC-1/output"
        self.console = "/home/cliu/Documents/SC-1/console"

        # delimiters for dividing a log into tokens
        self.delimiter = r'[ ,:()\[\]=|/\\{}\'\"<>]+'  # ,:()[]=|/\{}'"<>

        # two logs with editing distance less than distance_threshold
        # are considered to be similar
        self.distance_threshold = 0.2

        # how many chars are ignored from the beginning of a log
        # (because of the time-stamp, server-name, etc.)
        self.ignored_chars = 21

    def set_logfile(self, logfile):
        """
        Set the source log file (name/path) which is going to be analyzed.
        """
        self.logfile = logfile

    def set_delimiter(self, delimiter):
        """
        Set the delimiters (in regular expression)
        for dividing one log into tokens.
        """
        self.delimiter = delimiter

    def set_distance_threshold(self, distance_threshold):
        """
        Set the distance threshold 0 ~ 1 used for creating new cluster.
        The less the threshold is, the more similar two logs have to be
        if they want to be clustered together.
        """
        self.logfile = distance_threshold

    def set_ignored_chars(self, ignored_chars):
        """
        Set the ignored chars at the beginning of each log.
        """
        self.ignored_chars = ignored_chars

    def levenshtein_numpy(self, source, target):
        """
        Dynamic Programming algorithm, with the added optimization that only
        the last two rows of the dynamic programming matrix are needed for
        the computation. Vectorized version using NumPy.
        """
        if len(source) < len(target):
            return self.levenshtein_numpy(target, source)

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
        for one_source in source:
            # Insertion (target grows longer than source):
            current_row = previous_row + 1

            # Substitution or matching:
            # Target and source items are aligned, and either
            # are different (cost of 1), or are the same (cost of 0).
            current_row[1:] = np.minimum(current_row[1:],
                                         np.add(previous_row[:-1],
                                                target != one_source))

            # Deletion (target grows shorter than source):
            current_row[1:] = np.minimum(current_row[1:],
                                         current_row[0:-1] + 1)

            previous_row = current_row

        return float(previous_row[-1]) / float(max(len(source), len(target)))

    @classmethod
    def is_timestamp(cls, string):
        """
        Check whether input str is with time format like: Feb 11 05:22:51 .
        """
        try:
            time.strptime(string, '%b %d %H:%M:%S ')
            return True
        except ValueError:
            return False

    @classmethod
    def is_time(cls, string):
        """
        Check whether this is a time string.
        It supports most of the time format, more than is_timestamp()
        """
        try:
            timeparser(string)
            return True
        except ValueError:
            return False

    @classmethod
    def is_ipv4(cls, address):
        """
        Check whether this is a velid ipv4 address
        """
        try:
            socket.inet_pton(socket.AF_INET, address)
        except AttributeError:  # no inet_pton here, sorry
            try:
                socket.inet_aton(address)
            except socket.error:
                return False
            return address.count('.') == 3
        except socket.error:  # not a valid address
            return False

        return True

    @classmethod
    def is_ipv6(cls, address):
        """
        Check whether this is a velid ipv6 address
        """
        try:
            socket.inet_pton(socket.AF_INET6, address)
        except socket.error:  # not a valid address
            return False
        return True


    def is_ip_address(self, address):
        """
        Check whether this is a valid ip address (ipv4 or ipv6)
        """
        return self.is_ipv4(address) or self.is_ipv6(address)

    @classmethod
    def is_pci_address(cls, address):
        """
        Check whether this is a PCI address
        """
        pci_addr_pattern = r'(0000):([\da-fA-F]{2}):([\da-fA-F]{2}).(\d)'

        return re.search(pci_addr_pattern, address) is not None

    @classmethod
    def is_number(cls, number):
        """
        Check whether this is a number (int, long, float, hex)
        """
        try:
            float(number)  # for int, long, float
        except ValueError:
            try:
                int(number, 16) # for possible hex
            except ValueError:
                return False

        return True

    @classmethod
    def contain_hex(cls, string):
        """
        Check whether it contains hex
        """
        hex_pattern = r'0x[\da-fA-F]+'

        return re.search(hex_pattern, string) is not None


    # @classmethod
    def to_wildcard(self, tokens):
        """
        Replace number tokens and hex (0x...) tokens by wildcard symbol * .
        """
        # hex_pattern = r'0x[\da-fA-F]+'

        for i in range(0, len(tokens)):
            token = tokens[i]
            if (self.is_number(token) or
                    self.contain_hex(token) or
                    self.is_ip_address(token) or
                    self.is_pci_address(token)):
                tokens[i] = '*'

        return tokens

    def min_distance(self, added_line, cluster_dict):
        """
        Calculate the minimal distance between the log and all the clusters.
        Return the minimal distance and its index.
        """
        distance = {}
        added_line_tokens = self.to_wildcard(
            re.split(self.delimiter, added_line[self.ignored_chars:]))

        len_line = len(added_line_tokens)

        for cluster_num in cluster_dict:
            cluster = cluster_dict[cluster_num]
            cluster_line_tokens = self.to_wildcard(
                re.split(self.delimiter, cluster[0][self.ignored_chars:]))

            len_cluster = len(cluster_line_tokens)

            if (abs(len_cluster - len_line) / min(len_line, len_cluster) <
                    self. distance_threshold):
                dis_ratio = (float(editdistance.eval(cluster_line_tokens,
                                                     added_line_tokens)) /
                             float(min(len(added_line_tokens),
                                       len(cluster_line_tokens))))
            else:
                dis_ratio = float(1)

            distance[cluster_num] = dis_ratio

        # print distance
        mini = min(distance.iteritems(), key=lambda x: x[1])

        return mini[1], mini[0]

    def partition_by_command(self):
        """
        First partition the original logs based on their command type because:
        1. Dramatically reduce the computational time, especially plenty of
           time spent on levenshtein distance.
        2. Naturally, we should cluster logs starting with different command
           names into different clusters.
        """
        command_cluster = {}
        # pattern for extracting the command.
        # the command token could contain English letters, '-', '_' and '.'
        # example: rsyslogd, CMW, ERIC-RDA-Merged-Campaign,
        # mmas_syslog_control_setup.sh, etc.
        pattern = re.compile(r'([\w\-\_\./]+)([\[:])(.*)')
        current_num = 0

        with open(self.logfile) as in_file:
            added_line = in_file.readline()
            current_num = current_num + 1
            while not self.is_timestamp(added_line[:16]):
                added_line = in_file.readline()
                current_num = current_num + 1
            for line in in_file:
                current_num = current_num + 1
                # print current_num
                if not self.is_time(line[:16]):
                    added_line = added_line.rstrip() + ' | ' + line
                    # TODO(fluency03): use join()?
                    continue
                else:
                    # Do something for each log
                    command = re.match(pattern, added_line[21:]).group(1)
                    line_tokens = self.to_wildcard(
                        re.split(self.delimiter,
                                 added_line[self.ignored_chars:]))

                    length = len(line_tokens)

                    command_cluster.setdefault((command, length),
                                               [added_line]).append(added_line)

                    added_line = line

            # Add the last line
            # Do something for the last log
            command = re.match(pattern, added_line[21:]).group(1)
            line_tokens = self.to_wildcard(
                re.split(self.delimiter, added_line[self.ignored_chars:]))
            length = len(line_tokens)

            command_cluster.setdefault((command, length),
                                       [added_line]).append(added_line)

        return command_cluster

    def log_clustering(self, print_clusters=False):
        """
        Similarity checks and clustering after partitioning based on command.
        """
        command_cluster = self.partition_by_command()
        cluster_dict = {}

        for i in command_cluster:
            one_cluster_dict = {}
            cluster_length = len(cluster_dict)
            for line in command_cluster[i]:
                if not one_cluster_dict:
                    one_cluster_dict[cluster_length] = [line]
                else:
                    min_dis, min_index = self.min_distance(line,
                                                           one_cluster_dict)
                    if min_dis < self.distance_threshold:
                        one_cluster_dict[min_index].append(line)
                    else:
                        one_cluster_dict[len(cluster_dict) +
                                         cluster_length] = [line]

            cluster_dict.update(one_cluster_dict)

        if print_clusters:
            with open(self.console, 'w') as console_file:
                for i in cluster_dict:
                    console_file.write(str(i) + '\n')
                    for item in cluster_dict[i]:
                        console_file.write(item)

        return cluster_dict

    # def log_clustering_slow(self):
    #     """
    #     Log clustering without pre-partitioning based on command.
    #     This is much slower than logClusteringWithPrePartition().
    #     """
    #
    #     cluster_dict = {}
    #
    #     with open(self.logfile) as in_file:
    #         added_line = in_file.readline()
    #         while not self.is_timestamp(added_line[:16]):
    #             added_line = in_file.readline()
    #         for line in in_file:
    #             if not self.is_timestamp(line[:16]):
    #                 added_line = added_line.rstrip() + ' | ' + line
    #                 continue
    #             else:
    #                 # Do something for each log
    #                 if not cluster_dict:
    #                     cluster_dict[0] = [added_line]
    #                 else:
    #                     min_dis, min_index = self.min_distance(added_line,
    #                                                            cluster_dict)
    #                     if min_dis < self.distance_threshold:
    #                         cluster_dict[min_index].append(added_line)
    #                     else:
    #                         cluster_dict[len(cluster_dict)] = [added_line]
    #
    #                 added_line = line
    #
    #         # Add the last line
    #         # Do something for the last log
    #         min_dis, min_index = self.min_distance(added_line,
    #                                                cluster_dict)
    #         if min_dis < self.distance_threshold:
    #             cluster_dict[min_index].append(added_line)
    #         else:
    #             cluster_dict[len(cluster_dict)] = [added_line]
    #
    #     return cluster_dict


    # def token_collection(self, cluster):
    #     """
    #     Collect the unique tokens at each position of a log within a cluster
    #     """
    #     token_collection = {}
    #
    #     for item in cluster:
    #         line_tokens = self.to_wildcard(
    #             re.split(self.delimiter, item[self.ignored_chars:]))
    #
    #         len_line = len(line_tokens)
    #
    #         for i in range(0, len_line):
    #             token = line_tokens[i]
    #             one_collection[i].setdefault(token, 1)
    #
    #         token_collection[len_line] = one_collection
    #
    #     return token_collection



    def discover_template(self, print_clusters=False, print_templates=False):
        """
        Abstract the template representation from each of the clusters.
        """
        cluster_dict = self.log_clustering(print_clusters=print_clusters)
        template_dict = {}

        # for i in cluster_dict:
        #     token_collection = self.token_collection(cluster_dict[i])
        #     for length in token_collection:
        #         one_collection = token_collection[length]




        # TODO(fluency03): template representation



        if print_templates:
            with open(self.outfile, 'w') as out_file:
                for i in template_dict:
                    out_file.write(str(i) + '\n')
                    for item in template_dict[i]:
                        out_file.write(item)

        return template_dict








def main():
    """
    Main function
    """

    print "\nStart...\n"

    start_time = time.time()
    logfile = "/home/cliu/Documents/SC-1/messages.1"
    extractor = LogTemplateExtractor(logfile)
    # extractor.log_clustering_slow()
    # extractor.partition_by_command()
    # extractor.log_clustering()
    extractor.discover_template(print_clusters=True, print_templates=False)
    stop_time = time.time()

    print "\n--- %s seconds ---\n" % (stop_time - start_time)

    # ---------------------------- For debugging ---------------------------- #
    # list1 = ['a', 'b', 'c', 'd']
    # list2 = ['a', 'b', 'd', 'c', 'e']
    # print extractor.levenshtein_numpy(list1, list2)
    # str1 = ("SC-1 ecimswm: ActivateUpgradePackage::doAct`~ion: "
    #         "com<>pleted 2 of 2 procedures (100 percent)")
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

    # ---------------------------- For debugging ---------------------------- #

    print "\nStop...\n"


if __name__ == "__main__":
    # sys.stdout = open(console, 'w')
    main()

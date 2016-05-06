"""
This program is used to cluster log messages based on their similarity. The
similarity between two logs is measured based the editing distance between them.
For calculating editing distance, the basic unit of a log is not a character
but a token.

After the clustering, each of the generated cluster will be labeled with an
integer ID starting from 1. ID 0 represents a place with 'no-log'. The last ID
represents 'unknown-log' for further log matching. These IDs will be stored in
a dictionary for matching new logs.

The method 'levenshteinNumPy(source, target)'[1] implemented here is very slow
compared with the package 'editdistance'[2], since the source code of
'editdistance' is written in C++. However, if we would like to modify the
levenshtein algorithm by adding weights to different token classes
'levenshteinNumPy(source, target)' is easier to change the code.

Author: Chang Liu (fluency03)
Data: 2016-03-11

[1] Wikibooks: Algorithm Implementation/Strings/Levenshtein distance - Python:
https://en.wikibooks.org/wiki/Algorithm_Implementation/Strings/Levenshtein_distance#Python

[2] Fast implementation of the edit distance (Levenshtein distance) - C++:
https://github.com/aflc/editdistance

[3] The dateutil module provides powerful extensions to the standard datetime
module, available in Python.
https://github.com/dateutil/dateutil

"""


import glob
import cPickle as pickle
import os
import re
import socket
import time
import matplotlib.pyplot as plt
import numpy as np
import editdistance
from dateutil.parser import parse as timeparser


def is_timestamp(string):
    """
    Check whether input str is with time format like: Feb 11 05:22:51 .

    Aruguments:
        string: {string}, input string for time stamp check.
    """
    try:
        time.strptime(string, '%b %d %H:%M:%S ')
        return True
    except ValueError:
        return False


def is_time(string):
    """
    Check whether this is a time string.
    It supports most of the time format, more than is_timestamp().

    Aruguments:
        string: {string}, input string for time check.
    """
    try:
        timeparser(string)
        return True
    except ValueError:
        return False


def is_ipv4(address):
    """
    Check whether this is a velid ipv4 address.

    Aruguments:
        address: {string}, input string for ipv4 check.
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


def is_ipv6(address):
    """
    Check whether this is a velid ipv6 address.

    Aruguments:
        address: {string}, input string for ipv4 check.
    """
    try:
        socket.inet_pton(socket.AF_INET6, address)
    except socket.error:  # not a valid address
        return False
    return True


def contain_hex(string):
    """
    Check whether it contains hex values.

    Aruguments:
        string: {string}, input string for hex value check.
    """
    hex_pattern = r'0x[\da-fA-F]+'

    return re.search(hex_pattern, string) is not None

def is_ip_address(address):
    """
    Check whether this is a valid ip address (ipv4 or ipv6).

    Aruguments:
        address: {string}, input string for ip address(ipv4/ipv6) check.
    """
    return is_ipv4(address) or is_ipv6(address)


def is_pci_address(address):
    """
    Check whether this is a PCI address, like 0000:00:00.0

    Aruguments:
        address: {string}, input string for pci address check.
    """
    pci_addr_pattern = r'(0000):([\da-fA-F]{2}):([\da-fA-F]{2}).(\d)'

    return re.search(pci_addr_pattern, address) is not None


def is_number(number):
    """
    Check whether this is a number (int, long, float, hex) .

    Aruguments:
        number: {string}, input string for number check.
    """
    try:
        float(number)  # for int, long, float
    except ValueError:
        try:
            int(number, 16) # for possible hex
        except ValueError:
            return False

    return True


def to_wildcard(tokens):
    """
    Replace number tokens, hex (0x...) tokens, ip addresses and
    pci addresses by wildcard symbol * .

    Aruguments:
        tokens: {list}, a list of tokens.
    """
    for i in range(0, len(tokens)):
        token = tokens[i]
        if (is_number(token) or contain_hex(token) or
            is_ip_address(token) or is_pci_address(token)):
            tokens[i] = '*'
        else:
            # convert all digits in the token string into '0'
            tokens[i] = ''.join(('0' if char.isdigit() else char
                                 for char in token))
        # tokens[i] = token

    return tokens


def compare_two_tokens(token1, token2):
    """
    Compare two string tokens:
    if either of them is *, or they are equal, return True;
    else, return False.

    Aruguments:
        token1, token2: {string}, two tokens two be compared.
    """
    return token1 == '*' or token2 == '*' or token1 == token2


def check_directory(path):
    """
    Check whether the path/directory is existing. If not, create a new one.

    Aruguments:
        path: {string}, the given path/directory.
    """
    if not os.path.exists(path):
        print "Directory '%s' does not exist. Creat it... " %path
        os.makedirs(path)


class LogTemplateExtractor(object):
    """
    A log template extractor.

    Attributes:
        logfile_path: {string}, the path of log files to be analyzed.
        template_file: {string}, the output file for storing log templates.
        cluster_file: {string}, the output file for storing clustered logs.
        seqfile_path: {string}, the output file for storing log sequences.
        search_dict_file: {string}, the output file for storing search
            dictionary.
        delimiter_kept: {regex}, delimiters for dividing a log into tokens.
        cmd_regex: {regex}, regular expression for extracting command token
            from a log message.
        distance_threshold: {float}, two logs with editing distance less than
            this distance_threshold are considered to be similar. Default: 0.1.
        ignored_chars: {integer}, how many chars are ignored from the beginning
            of a log (because of the time-stamp, server-name, etc.) Default: 21.
        template_dict: {dictionary}, storing all the clustered log templates
            and their IDs.
        search_dict: {dictionary}, stroing tempalte IDs for new log matching.
    """
    def __init__(self, logfile_path):
        """
        Inits LogTemplateExtractor class.

        Aruguments:
            logfile_path: {string}, the path of log files to be be analyzed.
        """
        self.logfile_path = logfile_path
        self.template_file = "./template"
        self.cluster_file = "./clusters"
        self.search_dict_file = "./search_dict"
        self.seqfile_path = "./sequences/"

        # regex of delimiters for tokenization
        self.delimiter_kept = r'([\*\s,:()\[\]=|/\\{}\'\"<>\.\_\-])'

        # the command token could contain English letters, '-', '_', ' ' and '.'
        # example: rsyslogd, CMW, ERIC-RDA-Merged-Campaign,
        # mmas_syslog_control_setup.sh, JavaOaM install_imm_model.sh, etc.
        self.cmd_regex = r'([\w\-\_\./ \*]+)([\[:])(.*)'

        self.distance_threshold = 0.1
        self.ignored_chars = 21

        self.template_dict = {}
        self.search_dict = {}

    def set_logfile_path(self, logfile_path):
        """
        Set the source log file (name/path) which is going to be analyzed.

        Aruguments:
            logfile_path: {string}, the path of log files to be be analyzed.
        """
        self.logfile_path = logfile_path

    def set_seqfile_path(self, seqfile_path):
        """
        Set the sequence log file (name/path) which final sequence of the logs.

        Aruguments:
            seqfile_path: {string}, the path of log files to be be matched for
                generating sequences.
        """
        self.seqfile_path = seqfile_path

    def set_search_dict_file(self, search_dict_file):
        """
        Set the search_dict file (name/path) which search dictionary for
        new input log files.

        Aruguments:
            search_dict_file: {string}, the name/path of search dictionary file.
        """
        self.search_dict_file = search_dict_file

    def set_template_file(self, template_file):
        """
        Set the template log file (name/path) which tempalte IDs and
        their representations.

        Aruguments:
            template_file: {string}, the name/path of template/IDs file.
        """
        self.template_file = template_file

    def set_cluster_file(self, cluster_file):
        """
        Set the log cluster file (name/path) which template ID and the
        logs contained in each of the clusters.

        Aruguments:
            cluster_file: {string}, the name/path of clustered logs file.
        """
        self.cluster_file = cluster_file

    def set_delimiter(self, delimiter_kept):
        """
        Set the delimiters (in regex) for dividing one log into tokens.

        Aruguments:
            delimiter_kept: {regex}, delimiters for dividing a log into tokens.
        """
        self.delimiter_kept = delimiter_kept

    def set_distance_threshold(self, distance_threshold):
        """
        Set the distance threshold 0 ~ 1 used for creating new cluster.
        The less the threshold is, the more similar two logs have to be
        if they want to be clustered together.

        Aruguments:
            distance_threshold: {float}, distance_threshold to be set.
        """
        self.distance_threshold = distance_threshold

    def set_ignored_chars(self, ignored_chars):
        """
        Set the ignored chars at the beginning of each log.

        Aruguments:
            ignored_chars: {integer}, number of ignored chars in the beginning.
        """
        self.ignored_chars = ignored_chars

    def levenshtein_numpy(self, source, target):
        """
        Dynamic Programming algorithm, with the added optimization that only
        the last two rows of the dynamic programming matrix are needed for
        the computation. Vectorized version using NumPy.

        Aruguments:
            source, target: {list}, two lists of tokens to be compared.
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
        previous_row = np.arange(target.size + 1) # pylint: disable=E1101
        for item in source:
            # Insertion (target grows longer than source):
            current_row = previous_row + 1

            # Substitution or matching:
            # Target and source items are aligned, and either
            # are different (cost of 1), or are the same (cost of 0).
            current_row[1:] = np.minimum(current_row[1:],
                                         np.add(previous_row[:-1],
                                                target != item))

            # Deletion (target grows shorter than source):
            current_row[1:] = np.minimum(current_row[1:],
                                         current_row[0:-1] + 1)

            previous_row = current_row

        return float(previous_row[-1]) / float(max(len(source), len(target)))

    def tokenize(self, line):
        """
        Tokenize the line.

        Aruguments:
            line: {string}, one input log message.
        """
        return [t for t in re.split(self.delimiter_kept, line) if t is not '']

    def min_distance(self, added_line, one_cluster_dict):
        """
        Calculate the minimal distance between the log and all the sub-clusters
            from previous pre-partitioned cluster.

        Aruguments:
            added_line: {list}, a list of tokens.
            one_cluster_dict: {dictionary}, a dictionary for some clusters.
        Return the minimal distance and its index (key for cluster).
        """
        # dictionary of the distance between this log and
        # each of its compared clusters
        distance = {}

        len_line = len(added_line)

        for i in one_cluster_dict:
            cluster = one_cluster_dict[i]

            # the first log of this cluster represents this cluster
            cluster_line = cluster[0]
            len_cluster = len(cluster_line)

            # if the length difference is already beyond the distance threshold
            # there is no need to calculate the editing distance, and
            # the distance ratio will be set to 1
            if (abs(len_cluster - len_line) / min(len_line, len_cluster) <
                    self. distance_threshold):
                dis_ratio = (
                    float(editdistance.eval(cluster_line, added_line)) /
                    float(min(len(added_line), len(cluster_line))))
            else:
                dis_ratio = float(1)

            distance[i] = dis_ratio

        # find the minimal distance and its key value
        mini = min(distance.iteritems(), key=lambda x: x[1])

        return mini[1], mini[0]

    def add_log(self, added_line, command_cluster):
        """
        Add this log into partition, or create a new partition.

        Aruguments:
            added_line: {list}, a list of tokens.
            command_cluster: {dictionary}, a dictionary for certain cluster of
                logs with same command.
        Return the minimal distance and its index (key for cluster).
        """
        # pattern for extracting the command.
        cmd_pattern = re.compile(self.cmd_regex)

        # print added_line
        # extract command
        command = re.match(cmd_pattern,
                           added_line[self.ignored_chars:]).group(1)
        # tokenize the log message
        line_tokens = self.tokenize(added_line[self.ignored_chars:])
        # convert numbers, hexs, ip address, pci address to *
        line_tokens = to_wildcard(line_tokens)

        # get the length of this token list
        length = len(line_tokens)

        # if this cluster (command, length) existing,
        # append current log into its cluster;
        # if not, create a new cluster, key is (command, length),
        # initial value is [current log]
        command_cluster.setdefault(
            (command, length), []).append(line_tokens)

    def partition_by_command(self):
        """
        First partition the original logs based on their command type and the
        length of each log because:
        1. Dramatically reduce the computational time, especially plenty of
            time spent on levenshtein distance.
        2. Naturally, we should cluster logs starting with different command
            names into different clusters.
        3. The logs within one cluster sharing same length will make the next
            template extraction step easier.
        """
        # dictionary of the partitions divided based on
        # the tuple of command type and log length
        command_cluster = {}

        # keep track of the the number of each log
        # current_num = 0

        # log files
        logfiles = glob.glob(self.logfile_path)

        print "    |-Number of log files to be analyzed: %d" %len(logfiles)

        for logfile in logfiles:
            print "        " + logfile
            with open(logfile) as in_file:
                # read the first line
                added_line = in_file.readline()
                # current_num = current_num + 1

                # the real first line is the first log appearing with time-stamp
                while not is_timestamp(added_line[:16]):
                    added_line = in_file.readline()
                    # current_num = current_num + 1

                # read th following lines
                for line in in_file:
                    # current_num = current_num + 1

                    # if the current line is not with time-stamp, it will be
                    # added together to its previous logs until the the previous
                    # nearest log with time-stamp
                    if not is_time(line[:16]):
                        added_line = added_line.rstrip() + ' | ' + line
                        continue
                    else:
                        self.add_log(added_line, command_cluster)
                        # update added_line
                        added_line = line

                # Take the last line into account
                self.add_log(added_line, command_cluster)

        return command_cluster

    def log_clustering(self, print_clusters=False):
        """
        Similarity checks and clustering after partitioning based on command.
        Cluster ID starts from 1, all integers.

        Aruguments:
            print_clusters: {bool}, whether write the clusters into a file.
        """
        print "    |-Clustering logs..."

        # clusters based on command and log length
        command_cluster = self.partition_by_command()

        # dictionary of the log clusters
        cluster_dict = {}
        # keep track of the log cluster number
        cluster_num = 1

        for i in command_cluster:
            one_cluster_dict = {}
            for line in command_cluster[i]:
                if not one_cluster_dict:
                    one_cluster_dict[cluster_num] = [line]
                    cluster_num += 1
                else:
                    # get the minimal distance ratio and its index key
                    min_dis, min_key = self.min_distance(line, one_cluster_dict)

                    # if minimal distance ratio is less than the threshold,
                    # add this log into the cluster according to the index key;
                    # otherwise, create a new cluster
                    if min_dis < self.distance_threshold:
                        one_cluster_dict[min_key].append(line)
                    else:
                        one_cluster_dict[cluster_num] = [line]
                        cluster_num += 1

            # put all new clusters into the dictionary
            cluster_dict.update(one_cluster_dict)

        # print the clusters
        if print_clusters:
            print "    |-Write the clusters into %s ..." %self.cluster_file
            with open(self.cluster_file, 'w') as cluster_file:
                for i in cluster_dict:
                    cluster_file.write(str(i) + '\n')
                    for item in cluster_dict[i]:
                        cluster_file.write(''.join(item).rstrip() + '\n')
            print "    |-Write the clusters into %s.pkl ..." %self.cluster_file
            with open(self.cluster_file + '.pkl', 'w') as cluster_pkl_file:
                pickle.dump(cluster_dict, cluster_pkl_file)

        print "    |-Number of clusters generated: %d" %len(cluster_dict)

        return cluster_dict

    def log_template(self, cluster): # pylint: disable=R0201
        """
        Collect the unique tokens at each position of a log within a cluster.
        Update the positions where >1 unique tokens by wildcard *.
        Generate the template representation for this cluster.

        Aruguments:
            cluster: {dictionary}, a cluster of similar logs.
        """
        # the first log represents this cluster
        line_tokens = cluster[0]
        # get the length
        length = len(line_tokens)

        # a list of dictionaries represents at each of the token position
        # how many different tokens there are, and what are they
        token_collection = []
        for line in cluster:
            for i in range(0, length):
                token = line[i]
                if len(token_collection) > i:
                    token_collection[i].setdefault(token)
                else:
                    token_collection.append({token: None})

        # for positions sharing more than one unique token,
        # regard them as variables and convert them into *
        for i in range(0, length):
            if len(token_collection[i]) is not 1:
                line_tokens[i] = '*'

        return ''.join(line_tokens).rstrip() + '\n'

    def discover_template(self, print_clusters=False, print_templates=False):
        """
        Abstract the template representation from each of the clusters.

        Aruguments:
            print_clusters: {bool}, whether write the clusters into a file.
            print_templates: {bool}, whether write the templates into a file.
        """
        if os.path.isfile(self.cluster_file + '.pkl'):
            print "%s.pkl existing, loading it...\n" %self.cluster_file
            with open(self.cluster_file + '.pkl') as cluster_pkl_file:
                cluster_dict = pickle.load(cluster_pkl_file)
        else:
            # get the log cluster dictionary
            print "%s.pkl not existing, generate it...\n" %self.cluster_file
            cluster_dict = self.log_clustering(print_clusters=print_clusters)

        print "\n    |-Extracting templates..."

        # get each of the tempalte representations into the template_dict
        for i in cluster_dict:
            self.template_dict.setdefault(i, self.log_template(cluster_dict[i]))

        # print the template representations
        if print_templates:
            print "    |-Write the templates into %s ..." %self.template_file
            with open(self.template_file, 'w') as template_file:
                for i in self.template_dict:
                    template_file.write(str(i) + '\n')
                    for item in self.template_dict[i]:
                        template_file.write(item)
            print "    |-Write the templates into %s.pkl ..." %self.template_file
            with open(self.template_file + '.pkl', 'w') as template_pkl_file:
                pickle.dump(self.template_dict, template_pkl_file)

        print "    |-Number of tempaltes extracted: %d" %len(self.template_dict)

    def generate_search_dict(self, print_search_dict=False,
                             print_clusters=False, print_templates=False):
        """
        Generate the hashtable for matching new logs and ID them.

        Aruguments:
            print_search_dict: {bool}, whether write the search dictionaries
                into a file.
            print_clusters: {bool}, whether write the clusters into a file.
            print_templates: {bool}, whether write the templates into a file.
        """

        # Generate the template dictionary if it is empty.
        if os.path.isfile(self.template_file + '.pkl'):
            print "%s.pkl existing, loading it...\n" %self.template_file
            with open(self.template_file + '.pkl') as template_pkl_file:
                self.template_dict = pickle.load(template_pkl_file)
        else:
            print "%s.pkl not existing, generate it...\n" %self.template_file
            self.discover_template(print_clusters=print_clusters,
                                   print_templates=print_templates)

        print "\n    |-Generating the search dictionary..."

        # regex for extracting command
        cmd_pattern = re.compile(self.cmd_regex)

        # go through each of the log templates in the dictionary
        # and put their IDs into the search dictionary according to
        # the command and tokenized log length
        for id_ in self.template_dict:
            # get te tempalte representation
            tempalte = self.template_dict[id_]
            # print tempalte
            # get the command of thie template
            command = re.match(cmd_pattern, tempalte).group(1)

            # get the token list of this template
            tempalte_tokens = self.tokenize(tempalte)
            # get the length of this template
            length = len(tempalte_tokens)

            self.search_dict.setdefault((command, length),
                                        []).append(id_)

        # print the template search dictionary
        if print_search_dict:
            print ("    |-Writing the search dictionary into %s ..."
                   %self.search_dict_file)
            with open(self.search_dict_file, 'w') as search_dict_file:
                for i in self.search_dict:
                    search_dict_file.write('\n' + str(i) + '\n')
                    for item in self.search_dict[i]:
                        search_dict_file.write(str(item) + ' ')
            print ("    |-Writing the search dictionary into %s.pkl ..."
                   %self.search_dict_file)
            with open(self.search_dict_file + '.pkl',
                      'w') as search_dict_pkl_file:
                pickle.dump(self.search_dict, search_dict_pkl_file)

        print "    |-Template search dictionary generated!\n"

    def match_log(self, added_line, seq_file):
        """
        Match this log with the logs in search_dict.

        Aruguments:
            added_line: {string}, a line of log to be matched.
            seq_file: {file}, output sequence file.
        """
        # match flag
        is_matched = False

        # regex for extracting command
        cmd_pattern = re.compile(self.cmd_regex)
        # extract command
        command = re.match(cmd_pattern,
                           added_line[self.ignored_chars:]).group(1)

        # tokenize the log message
        line_tokens = self.tokenize(added_line[self.ignored_chars:])
        # convert numbers, hexs, ip address, pci address to *
        line_tokens = to_wildcard(line_tokens)
        # get the length of this token list
        length = len(line_tokens)

        # find this log in the search_dict
        if self.search_dict.has_key((command, length)):
            matched_list = self.search_dict[(command, length)]

            # compare the current new log to all tempaltes
            # in the selected matched_list
            for id_ in matched_list:
                # each of the tokenized template to be compared
                to_be_compared = self.tokenize(self.template_dict[id_])

                # put False into the compare result, if there is not-matching
                # between two tokens  at certain position
                compare_result = [False for a, b in zip(to_be_compared,
                                                        line_tokens)
                                  if not compare_two_tokens(a, b)]

                # if compare_result is empty, that means they are matched
                if not compare_result:
                    is_matched = True

                # if they are matched, ouput the template ID
                if is_matched:
                    seq_file.write(str(id_) + '\n')
                    # print str(current_num) + ' True'
                    break

            # if no match, that means this log is a new one
            # output the tmplate ID is '0', which means 'unknown'
            if not is_matched:
                seq_file.write('0\n')
                # print str(current_num) + ' False'


    def generate_sequence(self, new_logfile, print_search_dict=False,
                          print_clusters=False, print_templates=False):
        """
        Generate the log sequence based on previous generated templates and
        new input log files.
        Either: find the correct ID for each of the new log;
        Or: put the un-matched logs into the cluster '0', representing 'unknown'

        Aruguments:
            new_logfile: {string}, the path of log files to be matched for
                generating sequences.
            print_search_dict: {bool}, whether write the search dictionaries
                into a file.
            print_clusters: {bool}, whether write the clusters into a file.
            print_templates: {bool}, whether write the templates into a file.
        """

        # Generate the search_dict if it is empty.
        if os.path.isfile(self.search_dict_file + '.pkl'):
            print "%s.pkl existing, loading it...\n" %self.search_dict_file
            with open(self.search_dict_file + '.pkl') as search_dict_pkl_file:
                self.search_dict = pickle.load(search_dict_pkl_file)
        else:
            print "%s.pkl not existing, generate it...\n" %self.search_dict_file
            self.generate_search_dict(print_search_dict=print_search_dict,
                                      print_clusters=print_clusters,
                                      print_templates=print_templates)

        # current_num = 0

        print "Start to generate sequence."
        print "Writing the sequence into %s ..." %self.seqfile_path
        check_directory(self.seqfile_path)

        # log files
        new_logfiles = glob.glob(self.logfile_path)

        for new_logfile in new_logfiles:
            # print the template representations
            with open(new_logfile, 'r') as new_file:
                seqfile_path = self.seqfile_path + new_logfile.split("/")[-1]
                with open(seqfile_path, 'w') as seq_file:
                    print "        " + seqfile_path
                    added_line = new_file.readline()
                    # current_num = current_num + 1

                    # the real first line is the first log appearing with time-stamp
                    while not is_timestamp(added_line[:16]):
                        added_line = new_file.readline()
                        # current_num = current_num + 1

                    # read th following lines
                    for line in new_file:
                        # current_num = current_num + 1

                        # if the current line is not starting with time-stamp, it
                        # will be added together to its previous logs until the
                        # previous nearest log with time-stamp
                        if not is_time(line[:16]):
                            added_line = added_line.rstrip() + ' | ' + line
                            continue
                        else:
                            # match the log with search_dict
                            self.match_log(added_line, seq_file)
                            # update added_line
                            added_line = line

                    # Take the last line into account
                    self.match_log(added_line, seq_file)

        print "Sequece generated!\n"

    def generate_histogram(self):
        """
        Calculate the histogram for each of the generated sequence files.
        """
        print "Generate histogram...\n"
        # sequence files
        seq_files = glob.glob(self.seqfile_path + "*")

        for seq_file in seq_files:
            print "    " + seq_file
            with open(seq_file, 'r') as seqfile:
                sequence = [int(id_) for id_ in seqfile]
                hist, bin_edges = np.histogram(sequence, # pylint: disable=W0612
                                               bins=range(max(sequence)))
                plt.hist(hist, bins=range(max(sequence)))
                plt.xlim(0, 100)
                plt.ylim(0, 600)
                plt.savefig(seq_file.split("/")[-1])
                plt.clf()
                plt.cla()

    def plot_dots(self):
        """
        Plot curve for each of the generated sequence files.
        """
        print "Plot curve...\n"
        # sequence files
        seq_files = glob.glob(self.seqfile_path + "*")

        for seq_file in seq_files:
            print "    " + seq_file
            with open(seq_file, 'r') as seqfile:
                sequence = [int(id_) for id_ in seqfile]
                # t = np.arange(0, len(sequence), 1)
                plt.plot(sequence, 'r*')
                plt.xlim(0, 50000)
                plt.ylim(0, 3500)
                plt.savefig("curve_" + seq_file.split("/")[-1])
                plt.clf()
                plt.cla()

def main():
    """
    Main function
    """
    print "\nStart...\n"

    start_time = time.time()

    logfile_path = "./normal-logs/*"
    extractor = LogTemplateExtractor(logfile_path)
    extractor.set_template_file("./template")
    extractor.set_cluster_file("./clusters")
    extractor.set_seqfile_path("./sequences/")
    extractor.set_search_dict_file("./search_dict")

    extractor.generate_sequence(logfile_path, print_search_dict=True,
                                print_clusters=True, print_templates=True)

    # extractor.generate_histogram()

    # extractor.plot_dots()

    stop_time = time.time()

    print "Stop...\n"

    print "--- %s seconds ---\n" % (stop_time - start_time)


    # ---------------------------- For debugging ---------------------------- #


    # ---------------------------- For debugging ---------------------------- #



if __name__ == "__main__":
    main()

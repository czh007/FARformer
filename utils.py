# -*- coding:utf-8 -*-

import csv
import numpy as np


def findrange(icd, level, parient_child):
    for item in level:
        if '-' in item:
            tokens = item.split('-')
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:]) in range(int(tokens[0][1:]), int(tokens[1][1:]) + 1):
                    parient_child.append((item, icd))
                    return item
            else:
                # Not starting with E or V
                print(icd, tokens[0], tokens[1])
                if int(icd) in range(int(tokens[0]), int(tokens[1]) + 1):
                    parient_child.append((item, icd))
                    return item
        else:
            if icd.startswith('E') or icd.startswith('V'):
                if int(icd[1:]) == int(item[1:]):
                    return False
            else:
                # Not starting with E or V
                if int(icd) == int(item):
                    return False


def build_tree(filepath):
    level2 = ['001-009', '010-018', '020-027', '030-041', '042', '045-049', '050-059', '060-066', '070-079', '080-088',
              '090-099', '100-104', '110-118', '120-129', '130-136', '137-139', '140-149', '150-159', '160-165',
              '170-176',
              '176', '179-189', '190-199', '200-208', '209', '210-229', '230-234', '235-238', '239', '240-246',
              '249-259',
              '260-269', '270-279', '280-289', '290-294', '295-299', '300-316', '317-319', '320-327', '330-337', '338',
              '339', '340-349', '350-359', '360-379', '380-389', '390-392', '393-398', '401-405', '410-414', '415-417',
              '420-429', '430-438', '440-449', '451-459', '460-466', '470-478', '480-488', '490-496', '500-508',
              '510-519',
              '520-529', '530-539', '540-543', '550-553', '555-558', '560-569', '570-579', '580-589', '590-599',
              '600-608',
              '610-611', '614-616', '617-629', '630-639', '640-649', '650-659', '660-669', '670-677', '678-679',
              '680-686',
              '690-698', '700-709', '710-719', '720-724', '725-729', '730-739', '740-759', '760-763', '764-779',
              '780-789',
              '790-796', '797-799', '800-804', '805-809', '810-819', '820-829', '830-839', '840-848', '850-854',
              '860-869',
              '870-879', '880-887', '890-897', '900-904', '905-909', '910-919', '920-924', '925-929', '930-939',
              '940-949',
              '950-957', '958-959', '960-979', '980-989', '990-995', '996-999']
    level2_E = ['E000-E899', 'E000', 'E001-E030', 'E800-E807', 'E810-E819', 'E820-E825', 'E826-E829', 'E830-E838',
                'E840-E845', 'E846-E849', 'E850-E858', 'E860-E869', 'E870-E876', 'E878-E879', 'E880-E888', 'E890-E899',
                'E900-E909', 'E910-E915', 'E916-E928', 'E929', 'E930-E949', 'E950-E959', 'E960-E969', 'E970-E978',
                'E980-E989', 'E990-E999']
    level2_V = ['V01-V91', 'V01-V09', 'V10-V19', 'V20-V29', 'V30-V39', 'V40-V49', 'V50-V59', 'V60-V69', 'V70-V82',
                'V83-V84',
                'V85', 'V86', 'V87', 'V88', 'V89', 'V90', 'V91']

    allICDS = []  # Save all icds
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = [row[-1] for row in reader]
        for row in data:
            icds = row.split(';')
            allICDS.extend([icd for icd in icds if len(icd) > 0])

    allICDS_ = list(set(allICDS))
    allICDS_.sort(key=allICDS.index)

    # For each icd code in the EHR, find all its parent nodes and save them as (pariant,child)
    parient_child = []
    hier_icds = {}
    print('Loop start: for each icd code that appears in the EHR, find all its parent nodes')
    for icd in allICDS_:
        hier_icd = [icd]
        # First determine whether the icd contains E , for example: E939.58 or E824.1
        if icd.startswith('E'):
            # First, determine whether the decimal point is included.
            if '.' in icd:
                tokens = icd.split('.')
                # Then determine the number of decimal places
                if len(tokens[1]) == 1:
                    # After removing the decimal point, you will get the first parent node （E824,E824.1）
                    parient_child.append((tokens[0], icd))
                    hier_icd.insert(0, tokens[0])
                    # Find the range corresponding to E824
                    parient = findrange(tokens[0], level2_E, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)

                elif len(tokens[1]) == 2:
                    # After removing the decimal point to get will get three levels of the parent node
                    parient_child.append((icd[:-1], icd))  # （E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0], icd[:-1]))  # （E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient = findrange(tokens[0], level2_E, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)

        # First determine whether the icd contains V , for example: V85.54 or V86.0
        elif icd.startswith('V'):
            # First, determine whether the decimal point is included.
            if '.' in icd:
                tokens = icd.split('.')
                # Then determine the number of decimal places
                if len(tokens[1]) == 1:
                    # The first parent node is obtained after removing the decimal point (V86.0,V86)
                    parient_child.append((tokens[0], icd))
                    hier_icd.insert(0, tokens[0])
                    # Find the range corresponding to E824
                    parient = findrange(tokens[0], level2_V, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)


                elif len(tokens[1]) == 2:
                    # After removing the decimal point to get will get three levels of the parent node
                    parient_child.append((icd[:-1], icd))  # （E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0], icd[:-1]))  # （E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient = findrange(tokens[0], level2_V, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)

        else:
            # First, determine whether the decimal point is included.
            if '.' in icd:
                tokens = icd.split('.')
                # Then determine the number of decimal places
                if len(tokens[1]) == 1:
                    # After removing the decimal point, we get the first parent node (824,824.1)
                    parient_child.append((tokens[0], icd))
                    hier_icd.insert(0, tokens[0])
                    # Find the 824 corresponding range
                    parient = findrange(tokens[0], level2, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)

                elif len(tokens[1]) == 2:
                    # After removing the decimal point to get will get three levels of the parent node
                    parient_child.append((icd[:-1], icd))  # （E939.5，E939.58）
                    hier_icd.insert(0, icd[:-1])
                    parient_child.append((tokens[0], icd[:-1]))  # （E939，E939.5）
                    hier_icd.insert(0, tokens[0])
                    parient = findrange(tokens[0], level2, parient_child)
                    if parient != False:
                        hier_icd.insert(0, parient)
            else:  # No decimal point in disease
                # Find the 824 corresponding range
                parient = findrange(icd, level2, parient_child)
                if parient != False:
                    hier_icd.insert(0, parient)

        if icd not in hier_icds:
            hier_icds[icd] = hier_icd

    print('End of cycle')
    # Converting hierarchical labels to ids
    node2id = {}
    hier_labels_init = hier_icds.copy()

    for icd, hier_icd in hier_icds.items():
        # When hier_icdIds does not meet the length of 4, use the parent node for padding
        if len(hier_icd) < 5:
            hier_icd = hier_icd + [hier_icd[-1]] * (5 - len(hier_icd))
        hier_icds[icd] = hier_icd
        # Each node on the path is assigned a node
        for item in hier_icd:
            if item not in node2id:
                node2id[item] = len(node2id)
    hier_labels_init_new = {}
    for icd, hier_icd in hier_labels_init.items():
        icdId = node2id.get(icd)
        hier_icdIds = [node2id.get(item) for item in hier_icd]
        hier_labels_init_new[icdId] = hier_icdIds

    node2id['ROOT'] = len(node2id)
    # Convert character paths to id-type paths
    level0 = set()
    level1 = set()
    level2 = set()
    level3 = set()

    # Generate an adjacency matrix based on parient_child, which is used to easily find the children of each child node
    parient_child = []
    adj = np.zeros((len(node2id), len(node2id)))

    hier_dicts = {}

    print('Loop start: along the tree path, create adj')
    for icd, hier_icd in hier_icds.items():
        print('hier_icd:', hier_icd)
        icdId = node2id.get(icd)
        hier_icdIds = [node2id.get(item) for item in hier_icd]

        hier_icdIds.insert(0, node2id.get('ROOT'))
        hier_dicts[icdId] = hier_icdIds
        level0.add(hier_icdIds[1])
        level1.add(hier_icdIds[2])
        level2.add(hier_icdIds[3])
        level3.add(hier_icdIds[4])
        # Create adj along the tree path
        for i in range(len(hier_icdIds) - 1):
            for j in range(i + 1, i + 2):
                print('hier_icdIds[i]:', hier_icdIds[i])
                print('hier_icdIds[j]:', hier_icdIds[j])
                adj[hier_icdIds[i]][hier_icdIds[j]] = 1
                parient_child.append([hier_icdIds[i], hier_icdIds[j]])

    print('End of the cycle!')
    # Counting the number of the oldest children
    children_num = [len(np.argwhere(row)) for row in adj]
    max_children_num = max(len(level0), max(children_num))
    min_children_num = min(len(level0), min(children_num))
    print('max_childeren_num:', max_children_num)  # 72
    return parient_child, list(level0), list(level1), list(level2), list(
        level3), adj, node2id, hier_dicts, hier_labels_init_new, max_children_num


def build_brothers(filepath, node2id, hier_labels, parent_children_adj):
    brother_adj = np.zeros((len(node2id), len(node2id)))
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = [row[-1] for row in reader]
        for row in data:
            icds = row.split(';')
            icds = [node2id.get(icd) for icd in icds if len(icd) > 0]
            labels = [hier_labels.get(icd) for icd in icds]
            # Create a BROTHER relationship for each layer of each sample
            for level in range(1, 5):
                tmp = [row[level] for row in labels]
                for i in range(len(tmp) - 1):
                    for j in range(i + 1, len(tmp)):
                        # Determine whether tmp[i] and tmp[j] are siblings, i.e., whether they have the same father
                        brotherFlag = isbrother(tmp[i], tmp[j], parent_children_adj)
                        if brotherFlag:
                            brother_adj[tmp[i]][tmp[j]] += 1
                            brother_adj[tmp[j]][tmp[i]] += 1
    return brother_adj


def isbrother(label_a, label_b, parent_children_adj):
    parent_a = np.argwhere(parent_children_adj[:, label_a] > 0)
    parent_b = np.argwhere(parent_children_adj[:, label_b] > 0)
    if parent_a.any() == parent_b.any():
        brotherFlag = True
    else:
        brotherFlag = False
    return brotherFlag


def generate_graph(parient_child, node2id):
    import networkx as nx
    # Convert each edge in the parient-child to an id

    # Create diagrams based on relationships
    G = nx.Graph()
    G.add_nodes_from(node2id.values())
    G.add_edges_from(parient_child)
    print('number of nodes:', G.number_of_nodes())
    print('number of edges:', G.number_of_edges())
    return G


from collections import Counter


def get_label_probs(hier_labels, filepath, args):
    leafLabels = []
    with open(filepath, 'r') as f:
        reader = csv.reader(f)
        next(reader)
        data = [row[-1] for row in reader]
        for row in data:
            icds = row.split(';')
            leafLabels.extend([icd for icd in icds if len(icd) > 0])

    # According to the hierarchy tree each label on the path should be added 1
    allLabels = []
    for leaf in leafLabels:
        leafId = args.node2id.get(leaf)
        path = hier_labels.get(leafId)
        allLabels.append(path)
    # label_prob={}
    # For level_0 tags
    level_0 = [row[0] for row in allLabels]
    # Assign weights to level_0 tags
    label_count = Counter(level_0).most_common()
    print('label_count:', label_count)  # 第一跳节点

    label_weight_dict = {}
    baseNum = label_count[0][1]
    for key, value in label_count:
        # key=args.node2id.get(key)
        # probs = 1 * 1.0 / (value)
        label_weight_dict[key] = baseNum / value
    # Find the number of occurrences of the label expanded out by each label in level_0
    for a_0 in set(level_0):
        level_1 = [row[1] for row in allLabels if row[0] == a_0 and len(row) > 1]
        if len(level_1) > 0:
            # Assigning weights
            label_count = Counter(level_1).most_common()
            baseNum = label_count[0][1]
            for key, value in label_count:
                # key=args.node2id.get(key)
                # probs = 1 * 1.0 / (value)
                label_weight_dict[key] = baseNum / value

            # Find the number of occurrences of the label expanded out by each label in level_1
            for a_1 in set(level_1):
                level_2 = [row[2] for row in allLabels if len(row) > 2 and row[1] == a_1]
                if len(level_2) > 0:
                    # Assigning weights
                    label_count = Counter(level_2).most_common()
                    baseNum = label_count[0][1]
                    for key, value in label_count:
                        # key=args.node2id.get(key)
                        # probs = 1 * 1.0 / (value)
                        label_weight_dict[key] = baseNum / value

                    # Find the number of occurrences of the labels expanded out by each label in level_2
                    for a_2 in set(level_2):
                        level_3 = [row[3] for row in allLabels if (len(row) > 3 and row[2] == a_2)]
                        if len(level_3) > 0:
                            # Assigning weights
                            label_count = Counter(level_3).most_common()
                            baseNum = label_count[0][1]
                            for key, value in label_count:
                                # key=args.node2id.get(key)
                                # probs = 1 * 1.0 / (value)
                                label_weight_dict[key] = baseNum / value

    return label_weight_dict

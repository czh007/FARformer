# -*- coding:utf-8 -*-

import torch
import torch.nn.functional as F

from generator import action_space
import numpy as np
from collections import Counter
from sklearn.metrics import matthews_corrcoef,hamming_loss
import generator
import full_eval

def label_one_hot(patientLabels,class_num):
    # Convert labels from ID to multi-hot codes
    labels = []
    for row in patientLabels:
        temp = np.zeros(class_num)
        temp[row] = 1
        labels.append(temp)
    return np.array(labels)

def evaluate_topk_prob(gen_model,ehrs,hier_labels,train_flag=1):
    # 1. Get a representation of the electronic medical record
    ehrs = torch.Tensor(ehrs).long().to(gen_model.args.device)
    ehrRreps = gen_model.cnn(ehrs)  # [64,300]
    paths = [[gen_model.args.node2id.get('ROOT')] for i in range(len(ehrs))]
    pathRep = gen_model.pathEncoder(paths)  # [1,200]
    children, children_len = action_space(paths, gen_model.args)  # [64,2023] [64]
    golds_0 = []
    for labels in hier_labels:
        golds_0.append([row[1] for row in labels])
    golds_0 = label_one_hot(golds_0, class_num=len(gen_model.args.node2id))
    # hop == 0 # Each sample returns a different number of actions, so it is necessary to go through each sample individually later.
    actions = gen_model.ActionSelection.eval_act(gen_model.pathEncoder, ehrRreps, pathRep, children, golds_0, hop=0,
                                                 train_flag=train_flag)  # [64,229],[64]
    print('actions:', actions)
    all_paths = []
    for i in range(len(actions)):  # actions[i] is a list of variable length
        print('i:', i)
        samplePaths = []
        ehrRrep = ehrRreps[i].unsqueeze(0)
        for j in range(len(actions[i])):  # actions[i][j] are the possible actions for each predicted action, hop=0
            # Allocate the number of paths to be selected on each child node according to the score_hop0 probability
            # Execute the selected action to get the status and reward value
            paths_level_0 = [actions[i][j]]
            pathRep = gen_model.pathEncoder([paths_level_0])  # [1,200]

            # hop==1
            # Select action down according to the new state
            parient = [[actions[i][j]]]  # [3]
            children, children_len = action_space(parient, gen_model.args)  # [1,2023] [1]
            golds_1 = [row[2] for row in hier_labels[i]]
            golds_1 = label_one_hot([golds_1], class_num=len(gen_model.args.node2id))
            actions_hop1 = gen_model.ActionSelection.eval_act(gen_model.pathEncoder, ehrRrep, pathRep, children,
                                                              golds_1, hop=1, train_flag=train_flag)  # [1]

            for p in range(len(actions_hop1[0])):  # The possible actions when hop=1 is a list
                # Execute the selected action to get the status and reward value
                paths_level_1 = paths_level_0 + [actions_hop1[0][p]]

                # seqs_scores.append(candidate)
                pathRep = gen_model.pathEncoder([paths_level_1])  # [1,200]

                # hop==2
                # Select action down according to the new state
                parient = [actions_hop1[0][p]]
                children, children_len = action_space(parient, gen_model.args)  # [64,2023] [64]
                golds_2 = [row[3] for row in hier_labels[i]]
                golds_2 = label_one_hot([golds_2], class_num=len(gen_model.args.node2id))
                actions_hop2 = gen_model.ActionSelection.eval_act(gen_model.pathEncoder, ehrRrep, pathRep, children,
                                                                  golds_2, hop=2,
                                                                  train_flag=train_flag)  # [64,229],[64,1]

                for m in range(len(actions_hop2[0])):
                    # Execute the selected action to get the status and reward value
                    paths_level_2 = paths_level_1 + [actions_hop2[0][m]]

                    pathRep = gen_model.pathEncoder([paths_level_2])  # [1,200]

                    # hop==3
                    # Select action down according to the new state
                    parient = [actions_hop2[0][m]]
                    children, children_len = action_space(parient, gen_model.args)  # [64,2023] [64]

                    golds_3 = [row[4] for row in hier_labels[i]]
                    golds_3 = label_one_hot([golds_3], class_num=len(gen_model.args.node2id))
                    actions_hop3 = gen_model.ActionSelection.eval_act(gen_model.pathEncoder, ehrRrep, pathRep, children,
                                                                      golds_3, hop=3,
                                                                      train_flag=train_flag)  # [64,229],[64,1]

                    for n in range(len(actions_hop3[0])):
                        # Execute the selected action to get the status and reward value
                        paths_level_3 = paths_level_2 + [actions_hop3[0][n]]

                        samplePaths.append(paths_level_3)

        all_paths.append(samplePaths)

    # The resulting paths are the paths predicted by the agent for each sample
    return all_paths


def extend_sample(paths,actions,ehrs):
    selected_paths=[]
    ehrs_=[]
    hier_labels=[]
    # Select a path from the paths of each sample
    for i in range(len(paths)):
        # Perform data extensions
        for j in range(len(actions[i])):
            ehrs_.append(ehrs[i])
            selected_paths.append([actions[i][j]])
            hier_labels.append(paths[i])
    return ehrs_,selected_paths,hier_labels

def run_eval_step(gen_model,ehrs,hier_labels,train_flag):
    # Generate state,action from g_model
    # predPaths= evaluate_topk(gen_model,ehrs)
    predPaths = evaluate_topk_prob(gen_model, ehrs,hier_labels,train_flag)
    # Just return the predicted label path
    return predPaths


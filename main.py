# -*- coding:utf-8 -*-


import torch
import numpy as np
import argparse
import os
import data
from batcher import GenBatcher
from generator import Generator
import generator
import full_eval
from discriminator import Discriminator
import utils
import csv
import model_test_v2

from torch import autograd
import random
import pretrain_g_v2

# Specify the graphics card to run on
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

# Set random seeds
seed = 2023
np.random.seed(seed)
random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)

# Set some parameters
PARSER = argparse.ArgumentParser(description='the code of path generation')
PARSER.add_argument('-data_dir', '--data_dir', default='/home/ubuntu/RPGNet_MultiResCNN/data', type=str)
PARSER.add_argument('-vocab_size', '--vocab_size', default=50000, type=int,
                    help='Size of vocabulary')
PARSER.add_argument('-num_epochs', '--num_epochs', default=2000, type=int, help='num_epochs')
PARSER.add_argument('-batch_size', '--batch_size', default=256, type=int, help='batch size')
PARSER.add_argument('-lr', '--lr', default=0.00001, type=float, help='learning rate')
PARSER.add_argument('-hops', '--hops', default=4, type=int, help='number of hops')
PARSER.add_argument('-padded_len_ehr', '--padded_len_ehr', default=2000, type=int, help='padded length of ehr')

# CNN module parameters
PARSER.add_argument('-cnn_embedding_size', '--cnn_embedding_size', default=100, type=int,
                    help='the embedding size of CNN Network')
PARSER.add_argument('-num_filter_maps', '--num_filter_maps', default=100, type=int,
                    help='the num of filters of CNN Network')
PARSER.add_argument('-dropout_rate', '--dropout_rate', default=0.5, type=float, help='the dropout rate of  CNN Network')

# Parameters in the pathEncoder module
PARSER.add_argument('-node_embedding_size', '--node_embedding_size', default=300, type=int,
                    help='the embedding size of each node (GNN)')

# Parameters in the pathDecoder module
PARSER.add_argument('-path_hidden_size', '--path_hidden_size', default=100, type=int,
                    help='the hidden size of each path (LSTM)')

# Parameters in the DNQ module
PARSER.add_argument('-state_size', '--state_size', default=300, type=int, help='the size of state')

# D Parameters in the module
PARSER.add_argument('-d_hidden_size', '--d_hidden_size', default=300, type=int, help='the hidden size of D model')
PARSER.add_argument('-mode', '--mode', default='BCE', type=str, help='the type of loss')
PARSER.add_argument('-level', '--level', default='3', type=int,
                    help='different level of classification(range [0,1,2,3]')

# Parameters in MHR-CNN
PARSER.add_argument('-MODEL_DIR', type=str, default='./models')
PARSER.add_argument('-DATA_DIR', type=str, default='./data')
PARSER.add_argument('-MIMIC_3_DIR', type=str, default='./data/mimic3')
PARSER.add_argument('-MIMIC_2_DIR', type=str, default='./data/mimic2')

PARSER.add_argument("-data_path", type=str, default='./data/mimic3/train_full.csv')
PARSER.add_argument("-vocab", type=str, default='./data/mimic3/vocab.csv')
PARSER.add_argument("-Y", type=str, default='full', choices=['full', '50'])
PARSER.add_argument("-version", type=str, choices=['mimic2', 'mimic3'], default='mimic3')
PARSER.add_argument("-MAX_LENGTH", type=int, default=2500)

# model
PARSER.add_argument("-model", type=str, choices=['CNN', 'MultiCNN', 'ResCNN', 'MultiResCNN', 'bert_seq_cls'],
                    default='MultiResCNN')
PARSER.add_argument("-filter_size", type=str, default="3,5,9,15,19,25")
PARSER.add_argument("-conv_layer", type=int, default=1)
PARSER.add_argument("-embed_file", type=str, default='./data/mimic3/processed_full.embed')
PARSER.add_argument("-test_model", type=str, default=None)
PARSER.add_argument("-use_ext_emb", action="store_const", const=True, default=False)

# training
PARSER.add_argument("-n_epochs", type=int, default=2000)
PARSER.add_argument("-dropout", type=float, default=0.2)
PARSER.add_argument("-patience", type=int, default=10)
PARSER.add_argument("-weight_decay", type=float, default=0)
PARSER.add_argument("-criterion", type=str, default='prec_at_8', choices=['prec_at_8', 'f1_micro', 'prec_at_5'])
PARSER.add_argument("-gpu", type=int, default=-1, help='-1 if not use gpu, >=0 if use gpu')
PARSER.add_argument("-tune_wordemb", action="store_const", const=True, default=False)
PARSER.add_argument('-random_seed', type=int, default=1,
                    help='0 if randomly initialize the model, other if fix the seed')

# elmo
PARSER.add_argument("-elmo_options_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_options.json')
PARSER.add_argument("-elmo_weight_file", type=str, default='elmo_small/elmo_2x1024_128_2048cnn_1xhighway_weights.hdf5')
PARSER.add_argument("-elmo_tune", action="store_const", const=True, default=False)
PARSER.add_argument("-elmo_dropout", type=float, default=0)
PARSER.add_argument("-use_elmo", action="store_const", const=True, default=False)
PARSER.add_argument("-elmo_gamma", type=float, default=0.1)

# bert
PARSER.add_argument("-bert_dir", type=str)

args = PARSER.parse_args()
print(args)

args.device=('cuda:0' if torch.cuda.is_available() else 'cpu')
Variable = lambda *args, **kwargs: autograd.Variable(*args, **kwargs).to(args.device)


def pre_train_generator(gen_model, g_batcher, max_run_epoch):
    pretrain_g_v2.run_pre_train_step_v2(gen_model, g_batcher, max_run_epoch)


def pre_train_generator_batch(gen_model, ehrs, hier_labels):
    pretrain_g_v2.run_pre_train_step_batch(gen_model, ehrs, hier_labels)


# Training D model
def train_discriminator(g_model, d_model, ehrs, paths_n, ground_truths_hops, paths_p, mode=args.mode):
    loss = d_model.train_d_model(g_model, ehrs, paths_n, ground_truths_hops, paths_p, mode)
    return loss


def train_g(gen_model, d_model, log_action_episode_n, log_action_episode_p, rewards_episode_n, rewards_episode_p,
            ehrs_hops, paths_n):
    # Use real labels
    log_action_episode = torch.cat((log_action_episode_n, log_action_episode_p), dim=0)  # [128,4]
    rewards_episode = rewards_episode_n + rewards_episode_p
    rewards_episode = np.array(rewards_episode)  # [128,4]

    # Fully independent exploration of the label
    # log_action_episode=log_action_episode_n
    # rewards_episode=np.array(rewards_episode_n)

    import random
    index = [i for i in range(len(log_action_episode))]
    random.shuffle(index)
    log_action_episode = log_action_episode[index]
    rewards_episode = rewards_episode[index]

    # Update policy
    loss = gen_model.update_policy(d_model, rewards_episode, log_action_episode, ehrs_hops, paths_n)
    return loss


def evaluate(g_batcher, gen_model, writer, labelMask, flag='valid'):
    # Load all the data in the validation set
    if flag == 'train':
        train_flag = 1
    else:
        train_flag = 0
    batch_jaccard, batch_micro_p, batch_macro_p, batch_micro_r, batch_macro_r, batch_micro_f1, batch_macro_f1, batch_micro_auc_roc, batch_macro_auc_roc = [], [], [], [], [], [], [], [], []
    batches = g_batcher.get_batches(mode=flag)
    print('number of batches:', len(batches))
    for step in range(len(batches)):
        current_batch = batches[step]
        ehrs = [example.ehr for example in current_batch]
        hier_labels = [example.hier_labels for example in current_batch]
        with torch.no_grad():
            golds = []
            for labels in hier_labels:
                golds.append([row[-1] for row in labels])

            golds = label_one_hot(golds, class_num=len(gen_model.args.node2id))
            golds = np.array(golds)
            predHier_labels = model_test_v2.run_eval_step(gen_model, ehrs, hier_labels, train_flag)
            preds = []
            for sample in predHier_labels:
                pred = [row[-1] for row in sample]
                preds.append(pred)
            # Calculate the evaluation metrics for this one batch
            jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc, macro_auc_roc = full_eval.full_evaluate(
                preds, golds, len(gen_model.args.node2id), labelMask)
            print(
                'batchNum:{},jaccard:{:.4f},micro_p:{:.4f}, macro_p:{:.4f}, micro_r:{:.4f}, macro_r:{:.4f}, micro_f1:{:.4f}, macro_f1:{:.4f}, micro_auc_roc:{:.4f}, macro_auc_roc:{:.4f}'.format(
                    step, jaccard, micro_p, macro_p, micro_r, macro_r, micro_f1, macro_f1, micro_auc_roc,
                    macro_auc_roc))
            batch_jaccard.append(jaccard)
            batch_micro_p.append(micro_p)
            batch_macro_p.append(macro_p)
            batch_micro_r.append(micro_r)
            batch_macro_r.append(macro_r)
            batch_micro_f1.append(micro_f1)
            batch_macro_f1.append(macro_f1)
            batch_micro_auc_roc.append(micro_auc_roc)
            batch_macro_auc_roc.append(macro_auc_roc)
    avg_jaccard = sum(batch_jaccard) / len(batch_jaccard)
    avg_micro_p = sum(batch_micro_p) / len(batch_micro_p)
    avg_macro_p = sum(batch_macro_p) / len(batch_macro_p)
    avg_micro_r = sum(batch_micro_r) / len(batch_micro_r)
    avg_macro_r = sum(batch_macro_r) / len(batch_macro_r)
    avg_micro_f = sum(batch_micro_f1) / len(batch_micro_f1)
    avg_macro_f = sum(batch_macro_f1) / len(batch_macro_f1)
    avg_micro_auc_roc = sum(batch_micro_auc_roc) / len(batch_micro_auc_roc)
    avg_macro_auc_roc = sum(batch_macro_auc_roc) / len(batch_macro_auc_roc)

    print(
        'avg_jaccard:{:.4f},avg_micro_p:{:.4f}, avg_macro_p:{:.4f}, avg_micro_r:{:.4f}, avg_macro_r:{:.4f}, avg_micro_f1:{:.4f}, avg_macro_f1:{:.4f}, avg_micro_auc_roc:{:.4f}, avg_macro_auc_roc:{:.4f}'.format(
            avg_jaccard, avg_micro_p, avg_macro_p, avg_micro_r, avg_macro_r, avg_micro_f, avg_macro_f,
            avg_micro_auc_roc, avg_macro_auc_roc))
    return avg_micro_f, avg_macro_f


def label_one_hot(patientLabels, class_num):
    # Convert labels from ID to multi-hot codes
    labels = []
    for i, row in enumerate(patientLabels):
        temp = np.zeros(class_num)
        temp[row] = 1
        labels.append(temp)
    return labels


def main():
    print(torch.__version__)
    labels = 50
    samples = 50000  # Number of samples used Smaller numbers for easy debugging
    # Module 1: Data Preparation
    data_ = data.Data(args.data_dir, args.vocab_size, labels, samples)

    # Load the saved file
    import pickle
    if os.path.exists(os.path.join(args.data_dir, 'prepared_%d_sample_%d.pkl' % (labels, samples))):
        with open(os.path.join(args.data_dir, 'prepared_%d_sample_%d.pkl' % (labels, samples)), 'rb') as f:
            parient_children, brother_adj, level_0, level_1, level_2, level_3, adj, node2id, hier_dicts, hier_dicts_init, max_children_num = pickle.load(
                f)
        with open(os.path.join(args.data_dir, 'graph_%d_sample_%d.pkl' % (labels, samples)), 'rb') as f:
            graph = pickle.load(f)
    else:
        # Pre-processed data
        parient_children, level_0, level_1, level_2, level_3, adj, node2id, hier_dicts, hier_dicts_init, max_children_num = utils.build_tree(
            os.path.join(args.data_dir, 'filter_top_%d_sample%d.csv' % (labels, samples)))
        graph = utils.generate_graph(parient_children, node2id)
        brother_adj = utils.build_brothers(
            os.path.join(args.data_dir, 'filter_top_%d_sample%d.csv' % (labels, samples)), node2id, hier_dicts, adj)
        with open(os.path.join(args.data_dir, 'prepared_%d_sample_%d.pkl' % (labels, samples)), 'wb') as f:
            pickle.dump([parient_children, brother_adj, level_0, level_1, level_2, level_3, adj, node2id, hier_dicts,
                         hier_dicts_init, max_children_num], f)
        with open(os.path.join(args.data_dir, 'graph_%d_sample_%d.pkl' % (labels, samples)), 'wb') as f:
            pickle.dump(graph, f)

    args.node2id = node2id
    args.id2node = {id: node for node, id in node2id.items()}
    args.adj = torch.Tensor(adj).long().to(args.device)
    args.brother_adj = torch.Tensor(brother_adj).long().to(args.device)
    args.hier_dicts = hier_dicts
    args.level_0 = level_0
    args.level_1 = level_1
    args.level_2 = level_2
    args.level_3 = level_3
    args.level_0_mask = torch.Tensor(label_one_hot([args.level_0], len(args.node2id))).to(args.device)
    args.level_1_mask = torch.Tensor(label_one_hot([args.level_1], len(args.node2id))).to(args.device)
    args.level_2_mask = torch.Tensor(label_one_hot([args.level_2], len(args.node2id))).to(args.device)
    args.level_3_mask = torch.Tensor(label_one_hot([args.level_3], len(args.node2id))).to(args.device)
    args.max_children_num = max_children_num

    # Details of the batcher object
    g_batcher = GenBatcher(data_, args)

    # Module 2: Creating G models and pre-training G models
    # Generator object details
    gen_model_eval = Generator(args, data_, graph)
    gen_model_eval.to(args.device)

    # Module 3: Create D-model and pre-train D-model
    d_model = Discriminator(args, data_)
    d_model.to(args.device)

    # Pre-training every 10 rounds

    f = open('valid_result_%d_%d.csv' % (labels, samples), 'w')
    writer = csv.writer(f)
    writer.writerow(
        ['avg_micro_p', 'avg_macro_p', 'avg_micro_r,avg_macro_r', 'avg_micro_f1', 'avg_macro_f1', 'avg_micro_auc_roc',
         'avg_macro_auc_roc'])

    best_micro_f = 0.0
    best_macro_f = 0.0

    g_losses = []
    rewards = []
    d_losses = []

    for epoch in range(args.num_epochs):
        gen_model_eval.train()
        if epoch % 1 == 0:
            print('teacher forcing g model.....')
            pre_train_generator(gen_model_eval, g_batcher, max_run_epoch=1)

        epoch_g_loss = []
        epoch_reward = []
        epoch_d_loss = []
        batches = g_batcher.get_batches(mode='train')
        print('number of batches:', len(batches))
        for step in range(len(batches)):
            current_batch = batches[step]
            ehrs = [example.ehr for example in current_batch]
            hier_labels = [example.hier_labels.copy() for example in current_batch]

            gen_model_eval.train()
            reward, log_action_episode_n, log_action_episode_p, rewards_episode_n, rewards_episode_p, ehrs_hops, paths_n, ground_truths_hops, paths_p = generator.generate_samples(
                gen_model_eval, d_model, ehrs, hier_labels)
            # Take this batch of data tearcher force G network

            # Training G-networks
            g_loss = train_g(gen_model_eval, d_model, log_action_episode_n, log_action_episode_p, rewards_episode_n,
                             rewards_episode_p, ehrs_hops, paths_n)

            pre_train_generator_batch(gen_model_eval, ehrs, hier_labels)
            # Training D Network
            d_loss = train_discriminator(gen_model_eval, d_model, ehrs_hops, paths_n, ground_truths_hops, paths_p,
                                         mode=args.mode)
            epoch_d_loss.append(d_loss)
            print('batch_number:{}, reward:{:.4f},g_loss:{:.4f},d_loss:{:.4f}'.format(step, reward, g_loss, d_loss))
            epoch_g_loss.append(g_loss)
            epoch_reward.append(reward)

        # Get the average of all the loses of the batch in this epoch and add it to the loses
        g_losses.append(sum(epoch_g_loss) * 1.0 / len(epoch_g_loss))
        rewards.append(sum(epoch_reward) * 1.0 / len(epoch_reward))
        d_losses.append(sum(epoch_d_loss) * 1.0 / len(epoch_d_loss))

        if epoch % 1 == 0:
            gen_model_eval.eval()
            # Evaluating the model on a test set
            print('val result for epoch: %d.....................' % epoch)

            micro_f_train, macro_f_train = evaluate(g_batcher, gen_model_eval, writer, args.level_3, flag='train')

            # Evaluating the model on a test set
            print('test result for epoch: %d.....................' % epoch)
            micro_f_val, macro_f_val = evaluate(g_batcher, gen_model_eval, writer, args.level_3, flag='test')

            # Save the best micro model on the validation set
            if micro_f_val > best_micro_f:
                best_micro_f = micro_f_val
                torch.save(gen_model_eval.state_dict(),
                           os.path.join(args.data_dir, 'best_micro_gen_model_%d.pt' % labels))
                with open(os.path.join(args.data_dir, 'best_micro_threshold_%d.pkl' % labels), 'wb') as f:
                    pickle.dump(gen_model_eval.ActionSelection.best_threshold, f)

            # Save the best macro model on the validation set
            if macro_f_val > best_macro_f:
                best_macro_f = macro_f_val
                torch.save(gen_model_eval.state_dict(),
                           os.path.join(args.data_dir, 'best_macro_gen_model_%d.pt' % labels))
                with open(os.path.join(args.data_dir, 'best_macro_threshold_%d.pkl' % labels), 'wb') as f:
                    pickle.dump(gen_model_eval.ActionSelection.best_threshold, f)

    with open(os.path.join(args.data_dir, 'train_gLoss_dLoss_%d.csv' % labels), 'w') as f:
        writer = csv.writer(f)
        for l, d_l in zip(g_losses, d_losses):
            writer.writerow([l, d_l])

    with open(os.path.join(args.data_dir, 'train_reward_%d.csv' % labels), 'w') as f:
        writer = csv.writer(f)
        for item in rewards:
            writer.writerow([item])

if __name__ == '__main__':
    main()

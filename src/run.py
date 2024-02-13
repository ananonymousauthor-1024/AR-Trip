import numpy as np
import torch
import argparse
import time
import random
from tqdm import tqdm

from dataset import ARDataset
from trainer import ARTrainer
from utils import load_flickr_data, split_df_leave_one_out, poi_adjacent, poi_position

def main(opt):
    if opt.seed > 0:
        random.seed(opt.seed)
        torch.manual_seed(opt.seed)

    if opt.training_type == 'Normal':
        opt.Drifting = False
        opt.Guiding = False

    # define the calculator list
    all_f1_list = []
    all_pairs_f1_list = []
    all_repetition_list = []
    # Load and process data
    df, poi_dis_dict = load_flickr_data(dataset=opt.dataset)
    # counting the process time
    start_time = time.time()

    # Leave_one_out
    for split_index in tqdm(range(df.shape[0])):
        print(f'total trajectory is {df.shape[0]}, current index is {split_index}')

        train_df, test_df = split_df_leave_one_out(split_index, df, opt.seed)
        train_dataset = ARDataset(train_df)
        test_dataset = ARDataset(test_df)

        # initiate trainer
        venue_vocab_size = df['venue_ID'].explode().nunique() + 1
        hour_vocab_size = 24 + 1
        # Convert the venue_ID column to a list of lists
        venue_ids_lists = df["venue_ID"].tolist()
        # Calculate the maximum length of venue_ID
        max_length_venue_id = max(len(venue_ids) for venue_ids in venue_ids_lists)
        # counting the transfer matrix
        train_am = poi_adjacent(train_df, venue_vocab_size)
        train_pm, confidence = poi_position(train_df, venue_vocab_size, max_length_venue_id)

        if opt.decoding_type == 'Adapting':
            opt.confidence = confidence
        # feed the data to the trainer
        trainer = ARTrainer(train_dataset=train_dataset, eval_dataset=test_dataset, poi_dis_dict=poi_dis_dict,
                            lr=opt.lr, beta=opt.repetition_beta, confidence_score=opt.confidence,
                            data=opt.dataset, decode_type=opt.decoding_type, train_type=opt.training_type,
                            drifting=opt.Drifting, guiding=opt.Guiding, batch_size=opt.batch_size,
                            d_model=opt.d_model, num_encoder_layers=opt.num_encoder_layers,
                            num_epochs=opt.num_epochs, venue_vocab_size=venue_vocab_size,
                            hour_vocab_size=hour_vocab_size, max_length_venue_id=max_length_venue_id,
                            adjacent_matrix=train_am, position_matrix=train_pm)

        f1, pairs_f1, repetition = trainer.train()

        all_f1_list.append(f1)
        all_pairs_f1_list.append(pairs_f1)
        all_repetition_list.append(repetition)

    end_time = time.time()
    final_time = end_time - start_time
    print("the running time：%.2f seconds." % final_time)

    print("===" * 18)
    print("the final results...")

    # f1 and pairs_f1
    f1 = np.mean(all_f1_list)
    pairs_f1 = np.mean(all_pairs_f1_list)

    print(f'max f1 score: {f1}, max pairs_f1: {pairs_f1}')

    # repetition
    repetition = np.mean(all_repetition_list)
    print(f'total_repetition: {repetition}')

    with open(f'./results/{opt.dataset}.txt', 'a') \
            as file:
        file.write(f'============ Setting ============' + '\n')
        file.write(f'drifting: {opt.Drifting}' + '\n')
        file.write(f'guiding: {opt.Guiding}' + '\n')
        file.write(f'decoding type: {opt.decoding_type}' + '\n')
        file.write(f'training type: {opt.training_type}' + '\n')
        file.write(f'seed: {opt.seed}' + '\n')
        file.write(f'============ Result ============' + '\n')
        file.write(f'max f1-score: {f1}' + '\n')
        file.write(f'max pairs f1-score: {pairs_f1}' + '\n')
        file.write(f'repetition: {repetition}' + '\n')
        file.write(f'total process times: {final_time}' + '\n')

        file.write('============================================' + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # initial
    parser.add_argument('--seed', type=int, default=2023, help='manual seed')
    parser.add_argument('--dataset', type=str, default='Osak', help='dataset')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')  # 0.001
    parser.add_argument('--batch_size', type=int, default=4)
    parser.add_argument('--d_model', type=int, default=32)
    parser.add_argument('--num_encoder_layers', type=int, default=1)
    parser.add_argument('--num_epochs', type=int, default=50, help='number of epoch')
    # strategy
    parser.add_argument('--decoding_type', type=str, default='Adapting',
                        help='post-hoc decoding methods to fix the repetition problem. Candidate: Greedy, '
                             'Advanced-Greedy, Top-N, Top-NP, Adapting')
    parser.add_argument('--training_type', type=str, default='Penalty', help='strategies to fix the repetition '
                                                                             'problem. Candidate: Normal(only use '
                                                                             'CE-Loss), Penalty(using extra penalty '
                                                                             'loss drifting)')
    # hyperparameter
    parser.add_argument('--repetition_beta', type=float, default=1.0, help='the degree of repetition')  # 1.0
    parser.add_argument('--confidence', type=float, default=0.5, help='the re-scale degree')  # 0.5
    # ablation study
    parser.add_argument('--Drifting', action='store_true', help='ablation: using drifting')
    parser.add_argument('--Guiding', action='store_true', help='ablation: using guiding')
    args = parser.parse_args()

    # five repeat experiments
    # for args.seed in range(2020, 2025):
    # execute main program
    main(args)

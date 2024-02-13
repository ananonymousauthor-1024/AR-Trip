import logging
import pandas as pd
import torch
import numpy as np
import random
import torch.nn.functional as F


def load_data(dataset):
    """
    load data from csv file and then process it.
    (1) renumber the POI id, according to its frequency
    (2) renumber the user id, according to its frequency
    (3) number the category, according to its frequency
    (4) encode longitude and latitude (how?) -- current version: ignore
    -- maybe I can define multiple anchors, and calculate the distance between the current POI to them
    -- note: the longitude and latitude of a POI in (at) different trajectories (time) may vary
    -- note: it's ok, because we will concat the POI representation and spatial representation
    (5) encode UTC_time (how? need month, weekend / week days, day, hour?) -- current version: only hour, same to
    previous works
    (6) decompose data into multiple trajectories, each of which is a dict
    :param dataset:
    :return: grouped_df
    """
    path = f'./asset/data/{dataset}.csv'
    df = pd.read_csv(path)

    "(1) renumber the POI id, according to its frequency"
    poi_value_counts = df['venue_ID'].value_counts()
    poi_sorted_values = poi_value_counts.index.tolist()
    poi_renumber_dict = {value: i + 1 for i, value in enumerate(poi_sorted_values)}
    # print(poi_renumber_dict)
    df['venue_ID'] = df['venue_ID'].map(poi_renumber_dict)
    "(2) renumber the user id, according to its frequency"
    user_value_counts = df['user_ID'].value_counts()
    user_sorted_values = user_value_counts.index.tolist()
    user_renumber_dict = {value: i + 1 for i, value in enumerate(user_sorted_values)}
    # print(user_renumber_dict)
    df['user_ID'] = df['user_ID'].map(user_renumber_dict)
    "(3-1) number the category name, according to its frequency"
    cat_n_value_counts = df['venue_category_name'].value_counts()
    cat_n_sorted_values = cat_n_value_counts.index.tolist()
    cat_n_renumber_dict = {value: i + 1 for i, value in enumerate(cat_n_sorted_values)}
    # print(cat_n_renumber_dict)
    df['venue_category_name'] = df['venue_category_name'].map(cat_n_renumber_dict)
    "(3-2) number the category id, according to its frequency"
    cat_i_value_counts = df['venue_category_ID'].value_counts()
    cat_i_sorted_values = cat_i_value_counts.index.tolist()
    cat_i_renumber_dict = {value: i + 1 for i, value in enumerate(cat_i_sorted_values)}
    # print(cat_i_renumber_dict)
    df['venue_category_ID'] = df['venue_category_ID'].map(cat_i_renumber_dict)
    "(4)"

    "(5) encode UTC (1~24)"
    df['UTC_time'] = pd.to_datetime(df['UTC_time'])
    df['hour'] = df['UTC_time'].dt.hour
    "(6) merge the trajectory and time by traj_id"
    grouped_df = df.groupby('traj_id').agg({
        'user_ID': 'first',  # Keep the first user ID for each trajectory
        'venue_ID': list,  # Aggregate the POIs into a list for each trajectory
        'hour': list,  # Optional: Aggregate the UTC times into a list for each trajectory
        'latitude': list,
        'longitude': list
    }).reset_index()
    return grouped_df


def split_df_train_eval_test(df, random_seed, ratio):
    """
    split data into train_df, eval_df and test_df

    Parameters:
    :param df: The DataFrame to be split.
    :param random_seed: Seed for random number generator.
    :param ratio: The ratio in which to split the data.

    Returns:
    :return:Three DataFrames: train_df, eval_df, and test_df.
    """
    if not isinstance(ratio, tuple):
        raise TypeError("ratio must be a tuple.")
    if ratio[0] + ratio[1] + ratio[2] != 1:
        raise AssertionError('ratio should be normalized.')

    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Calculate the sizes for train, eval, and test datasets
    total_size = len(df)
    train_size = int(ratio[0] * total_size)
    eval_size = int(ratio[1] * total_size)
    # Split the data into train, eval, and test datasets
    train_df = df[:train_size]
    eval_df = df[train_size:train_size + eval_size]
    test_df = df[train_size + eval_size:]
    return train_df, eval_df, test_df


def load_flickr_data(dataset):
    """
    load dataset and process like "load_data"
    :param dataset:
    :return: grouped_df, renumber_poi_distance_dict: {re_poi: (lon, lat)}
    """
    path = f'./asset/data/{dataset}.csv'
    df = pd.read_csv(path)

    "(1) renumber the POI id, according to its frequency"
    poi_value_counts = df['venue_ID'].value_counts()
    poi_sorted_values = poi_value_counts.index.tolist()
    poi_renumber_dict = {value: i + 1 for i, value in enumerate(poi_sorted_values)}
    # print(poi_renumber_dict)
    df['venue_ID'] = df['venue_ID'].map(poi_renumber_dict)

    "(2) renumber the user id, according to its frequency"
    user_value_counts = df['user_ID'].value_counts()
    user_sorted_values = user_value_counts.index.tolist()
    user_renumber_dict = {value: i + 1 for i, value in enumerate(user_sorted_values)}
    # print(user_renumber_dict)
    df['user_ID'] = df['user_ID'].map(user_renumber_dict)

    "(3) number the category name, according to its frequency"
    cat_n_value_counts = df['venue_category_name'].value_counts()
    cat_n_sorted_values = cat_n_value_counts.index.tolist()
    cat_n_renumber_dict = {value: i + 1 for i, value in enumerate(cat_n_sorted_values)}
    # print(cat_n_renumber_dict)
    df['venue_category_name'] = df['venue_category_name'].map(cat_n_renumber_dict)

    "(4) encode UTC (1~24)"
    df['UTC_time'] = pd.to_datetime(df['UTC_time'])
    df['hour'] = df['UTC_time'].dt.hour

    "(5) merge the trajectory and time by traj_id"
    grouped_df = df.groupby('traj_ID').agg({
        'user_ID': 'first',  # Keep the first user ID for each trajectory
        'venue_ID': list,  # Aggregate the POIs into a list for each trajectory
        'hour': list,  # Optional: Aggregate the UTC times into a list for each trajectory
        'latitude': list,  # each poi in certain trajectory
        'longitude': list  # each poi in certain trajectory
    }).reset_index()

    "(6) add renumber_poi_distance_dict to count the distance, also add padded number 0"
    renumber_poi_distance_df = df[['venue_ID', 'longitude', 'latitude']].drop_duplicates().sort_values(
        by='venue_ID').reset_index(drop=True)
    venue_data_zero = {'venue_ID': 0, 'longitude': 0.0, 'latitude': 0.0}
    venue_data_zero_df = pd.DataFrame([venue_data_zero])
    renumber_poi_distance_df = pd.concat([venue_data_zero_df, renumber_poi_distance_df], ignore_index=True)

    renumber_poi_distance_dict = renumber_poi_distance_df.groupby('venue_ID').apply(
        lambda x: (x['longitude'].iloc[0], x['latitude'].iloc[0])).to_dict()

    return grouped_df, renumber_poi_distance_dict


def split_df_leave_one_out(split_index, df, random_seed):
    """
    split data into train_df, test_df

    :param split_index:
    :param df:
    :param random_seed:
    :return:
    """
    # Shuffle the data
    df = df.sample(frac=1, random_state=random_seed).reset_index(drop=True)
    # Calculate the sizes for train and test datasets

    train_df = df[df.index != split_index]
    test_df = df[df.index == split_index]

    return train_df, test_df


# Padding Function
def pad_sequences(sequences, pad_value=0, max_len=None):
    padded_sequences = torch.nn.utils.rnn.pad_sequence(sequences, batch_first=True, padding_value=pad_value)
    if max_len:
        if padded_sequences.size(1) < max_len:
            padded_sequences = torch.nn.functional.pad(padded_sequences, (0, max_len - padded_sequences.size(1)))
    return padded_sequences


def collate_fn(batch):
    masked_venue_ids, masked_hour_ids, venue_ids, hour_ids, masked_latitudes, masked_longitudes = zip(*batch)
    padded_venue_ids = pad_sequences(venue_ids, pad_value=0)
    padded_hour_ids = pad_sequences(hour_ids, pad_value=0)
    masked_padded_latitudes = pad_sequences(masked_latitudes, pad_value=0)
    masked_padded_longitudes = pad_sequences(masked_longitudes, pad_value=0)
    masked_padded_venue_ids = pad_sequences(masked_venue_ids, pad_value=0)
    masked_padded_hours = pad_sequences(masked_hour_ids, pad_value=0)

    return masked_padded_venue_ids, masked_padded_hours, padded_venue_ids, padded_hour_ids, masked_padded_latitudes, \
        masked_padded_longitudes


def get_logger(filename, level=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[level])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    fh.setLevel(level_dict[level])
    logger.addHandler(fh)

    return logger

# ============================= metrics =========================== #
def f1_score(target, predict, noloop=False):
    """
    Compute F1 Score for recommended trajectories
    :param target: the actual trajectory
    :param predict: the predict trajectory
    :param noloop:

    :return: f1
    """
    assert (isinstance(noloop, bool))
    assert (len(target) > 0)
    assert (len(predict) > 0)

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1

    f1 = 2 * precision * recall * 1.0 / denominator

    return f1


def pairs_f1_score(target, predict):
    """
    Compute Pairs_F1 Score for recommended trajectories
    :param target:
    :param predict:
    :return: pairs_f1
    """
    # Check if number of elements > 0
    assert target.numel() > 0
    n = target.numel()
    nr = predict.numel()
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = dict()
    for i, poi in enumerate(target):
        order_dict[poi.item()] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i].item()
        for j in range(i + 1, nr):
            poi2 = predict[j].item()
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        pairs_f1 = 0
    else:
        pairs_f1 = 2. * precision * recall / (precision + recall)

    return pairs_f1


def calc_dist_vec(longitudes1, latitudes1, longitudes2, latitudes2):
    """Calculate the distance (unit: km) between two places on earth, vectorised"""
    # convert degrees to radians
    lng1 = np.radians(longitudes1)
    lat1 = np.radians(latitudes1)
    lng2 = np.radians(longitudes2)
    lat2 = np.radians(latitudes2)
    radius = 6371.0088  # mean earth radius, en.wikipedia.org/wiki/Earth_radius#Mean_radius

    # The haversine formula, en.wikipedia.org/wiki/Great-circle_distance
    dlng = np.fabs(lng1 - lng2)
    dlat = np.fabs(lat1 - lat2)
    dist = 2 * radius * np.arcsin(np.sqrt(
        (np.sin(0.5 * dlat)) ** 2 + np.cos(lat1) * np.cos(lat2) * (np.sin(0.5 * dlng)) ** 2))
    return dist


def count_repetition_percentage(input_data):
    # if list
    if isinstance(input_data, list):
        unique_items = set(input_data)
    # if tensor
    elif hasattr(input_data, 'numpy'):
        unique_items = set(input_data.numpy().tolist())
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total_items = len(input_data)
    repetition_items_count = total_items - len(unique_items)
    repetition_ratio = repetition_items_count / total_items

    return repetition_ratio


# ===================== decoding methods =========================== #
# Advanced-Greedy:
def find_duplicates_and_indices(input_tensor):

    duplicates_dict = {}
    input_tensor = input_tensor.tolist()
    for index, value in enumerate(input_tensor):
        if input_tensor.count(value) > 1:
            if value not in duplicates_dict:
                duplicates_dict[value] = [index, ]
            else:
                duplicates_dict[value] += [index, ]

    return duplicates_dict

def advanced_greedy_recommendation(batch_candidate, batch_similarity):

    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        if len(top_candidates[batch]) == len(np.unique(top_candidates[batch])):
            pass
        else:

            position_top_k = [0] * top_candidates.shape[1]

            while len(top_candidates[batch]) != len(np.unique(top_candidates[batch])):

                repetition_dict = find_duplicates_and_indices(top_candidates[batch])

                for key in repetition_dict.keys():
                    confidence_list = []
                    for item in repetition_dict[key]:
                        confidence = batch_similarity[batch, item, position_top_k[item]].item()
                        confidence_list.append(confidence)

                    max_item = repetition_dict[key][confidence_list.index(max(confidence_list))]
                    left_item_list = list(filter(lambda x: x != max_item, repetition_dict[key]))
                    for left_item in left_item_list:

                        position_top_k[left_item] += 1
                        top_candidates[batch, left_item] = batch_candidate[batch, left_item, position_top_k[left_item]]

    return top_candidates

# Top-N and Top-NP method:
def random_choice_by_probability(probability_list):

    cumulative_probabilities = []
    cumulative_prob = 0
    for prob in probability_list:
        cumulative_prob += prob
        cumulative_probabilities.append(cumulative_prob)

    random_number = random.random()

    for i, cumulative_prob in enumerate(cumulative_probabilities):
        if random_number <= cumulative_prob:
            return i


def select_top_p_indices(probabilities, threshold=0.8):

    sorted_indices = np.argsort(probabilities)[::-1]  # re-order the probability
    cumulative_prob = 0.0
    selected_indices = []

    for idx in sorted_indices:
        cumulative_prob += probabilities[idx]
        selected_indices.append(idx)
        if cumulative_prob >= threshold:
            break

    return selected_indices[-1]

def top_n_recommendation(batch_candidate, batch_similarity, confidence=1):

    # the top_n method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):

            # print(batch_similarity[batch, middle_index])
            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] * confidence, dim=0)
            # print(batch_similarity[batch, middle_index])
            new_top_k_index = random_choice_by_probability(batch_similarity[batch, middle_index].tolist())
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_k_index]

    return top_candidates  # [b,l]


def top_np_recommendation(batch_candidate, batch_similarity, confidence=1, threshold=0.8):

    # the top_np method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):

            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] * confidence, dim=0)

            top_p_indices = select_top_p_indices(batch_similarity[batch, middle_index].tolist(), threshold)
            batch_similarity[batch, middle_index, :(top_p_indices+1)] = \
                F.softmax(batch_similarity[batch, middle_index, :(top_p_indices+1)] * confidence, dim=0)

            batch_similarity[batch, middle_index, (top_p_indices+1):] = torch.tensor(0)

            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates  # [b,l]

def ad_top_np_recommendation(batch_candidate, batch_similarity, confidence, threshold=0.8):

    # the top_np method to recommend trajectory
    top_candidates = batch_candidate[:, :, 0].cpu()  # [b,l]
    batch_similarity = batch_similarity.cpu()

    for batch in range(batch_candidate.shape[0]):
        for middle_index in range(batch_candidate.shape[1]):

            batch_similarity[batch, middle_index] = F.softmax(batch_similarity[batch, middle_index] *
                                                              confidence[middle_index], dim=0)

            top_p_indices = select_top_p_indices(batch_similarity[batch, middle_index].tolist(), threshold)
            batch_similarity[batch, middle_index, :(top_p_indices+1)] = \
                F.softmax(batch_similarity[batch, middle_index, :(top_p_indices+1)], dim=0)

            batch_similarity[batch, middle_index, (top_p_indices+1):] = torch.tensor(0)

            batch_probability_list = batch_similarity[batch, middle_index].tolist()
            nonzero_probability_list = [x for x in batch_probability_list if x != 0]

            new_top_p_index = random_choice_by_probability(nonzero_probability_list)
            top_candidates[batch, middle_index] = batch_candidate[batch, middle_index, new_top_p_index]

    return top_candidates

# ==================================== evaluate ======================================== #
# find best recommendation among the same epoch
def count_best_index(lst1, lst2, lst3):

    best_indices = [i for i, value in enumerate(lst1) if value ==
                    max(lst1)]
    sublist_2 = [lst2[index] for index in best_indices]

    best_indices = [value for i, value in enumerate(best_indices) if sublist_2[i] ==
                    max(sublist_2)]

    sublist_3 = [lst3[index] for index in best_indices]
    best_indices = [value for i, value in enumerate(best_indices) if sublist_3[i] ==
                    min(sublist_3)]
    best_index = best_indices[-1]

    return best_index

# ==================== transfer matrix ==============================

# generate transfer matrix
def poi_adjacent(dataset, poi_size):
    traj_list = dataset['venue_ID'].tolist()

    AM = np.zeros((poi_size, poi_size), dtype=int)
    for traj in traj_list:
        for index in range(len(traj)-1):
            curr_poi = traj[index]
            next_poi = traj[index + 1]
            AM[curr_poi][next_poi] += 1  # [v,v] pi -> pi+1

    row_sums = AM.sum(axis=1)
    AM = AM / row_sums[:, np.newaxis]
    AM[np.isnan(AM)] = 0

    return AM.astype(np.float32)

def poi_position(dataset, poi_size, max_length):
    traj_list = dataset['venue_ID'].tolist()

    PM = np.zeros((poi_size, max_length))
    for traj in traj_list:
        for index in range(len(traj)):
            PM[traj[index]][index] += 1  # [v,l_max]

    row_sums = PM.sum(axis=1)
    PM = PM / row_sums[:, np.newaxis]
    PM[np.isnan(PM)] = 0

    zero_counts = np.count_nonzero(PM == 0, axis=0)
    total_points = PM.shape[0]
    confidence = zero_counts / total_points
    confidence = [min(0.5, val) for val in confidence]
    # print(PM)
    return PM.astype(np.float32), confidence


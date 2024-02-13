import argparse
import os
import pandas as pd
from datetime import datetime

"""
preprocess for flickr dataset
filter sub-trajectory less than 3 POIs
"""


def main(opt):
    if opt.dataset not in ['Osak', 'Toro', 'Edin', 'Melb', 'Glas']:
        raise AssertionError('Unsupported dataset!!!')
    # exist python file path
    current_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'dataset_flickr')
    meta_data_poi_filename = f'poi-{opt.dataset}.csv'
    meta_data_traj_filename = f'traj-{opt.dataset}.csv'
    meta_data_poi_path = os.path.join(data_dir, meta_data_poi_filename)
    meta_data_traj_path = os.path.join(data_dir, meta_data_traj_filename)

    poi_df = pd.read_csv(meta_data_poi_path)
    poi_df.columns = ['venue_ID', 'venue_category_name', 'longitude', 'latitude']
    traj_df = pd.read_csv(meta_data_traj_path)
    traj_df.columns = ['user_ID', 'traj_ID', 'venue_ID', 'UTC_time', 'end_time', 'photo', 'traj_Len', 'Duration']
    """step 1: filter the useful info in df_traj"""
    traj_df = traj_df.loc[:, ['user_ID', 'traj_ID', 'venue_ID', 'UTC_time', 'traj_Len']]

    """step 2: filter sub-trajectory"""
    traj_df = traj_df[traj_df['traj_Len'] >= 3]

    """step 3: convert the local time to the utc_time, sort value by time, and renumber the trajID"""
    traj_df['UTC_time'] = pd.to_datetime(traj_df['UTC_time'], unit='s')
    traj_df.sort_values(by=['traj_ID', 'UTC_time'], inplace=True)
    traj_df['traj_ID'] = pd.factorize(traj_df['traj_ID'])[0] + 1
    """step 4: merge the poi and traj file"""
    merge_df = traj_df.merge(poi_df, on='venue_ID', how='left')
    """step 5: save the results to the file"""
    merge_df.to_csv(f'../data/{opt.dataset}.csv', index=False)
    print(merge_df)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='Osak', help='dataset')
    args = parser.parse_args()

    main(args)

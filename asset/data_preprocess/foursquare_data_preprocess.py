import argparse
import os
from rich.console import Console
from rich.progress import track
import pandas as pd
from datetime import datetime
from rich.progress import Progress
from functools import reduce
from src.utils import *

"""
for NYC and TKY

several steps:
    (0) drop duplicates
    (1) filter POI occurred less than 5 times
    (2) generate trajectory for each user
    (3) split sub-trajectory via 6-hour
    (4) filter duplicates poi (same poi, different time -- self-loop trip)
    (5) filter sub-trajectory less than 3 POIs
    (6) filter user who has less than 5 trajectories
"""


def main(opt):
    """logger"""
    console = Console(color_system='256', style=None)

    """dataset selection"""
    meta_data_path, encoding_type = None, None

    current_path = os.path.dirname(os.path.abspath(__file__))
    data_dir = os.path.join(os.path.dirname(os.path.dirname(current_path)), 'data', 'dataset_tsmc2014')

    if opt.dataset == 'NYC':
        meta_data_path = os.path.join(data_dir, 'dataset_TSMC2014_NYC.txt')
    elif opt.dataset == 'TKY':
        meta_data_path = os.path.join(data_dir, 'dataset_TSMC2014_TKY.txt')
    else:
        assert AssertionError('Unsupported dataset!!!')

    """load data and add data head"""
    df = pd.read_csv(meta_data_path, delimiter='\t', header=None, encoding="ISO-8859-1")
    df.columns = ['user_ID', 'venue_ID', 'venue_category_ID', 'venue_category_name',
                  'latitude', 'longitude', 'timezone_offset', 'UTC_time']
    console.log(f'[bold cyan]pre-processing [bold red]{opt.dataset}[bold cyan] dataset, '
                f'total [bold red]{len(df)}[bold cyan] check-ins')

    """time conversion"""
    df['UTC_time'] = df['UTC_time'].apply(lambda x: datetime.strptime(x, "%a %b %d %H:%M:%S %z %Y"))
    df['UTC_time'] = pd.to_datetime(df['UTC_time'])

    """step 0: drop duplicates"""
    df_no_duplicates = df.drop_duplicates()
    df_no_duplicates = df_no_duplicates.reset_index(drop=True)
    console.log(f'[bold cyan]after [bold red]dropping[bold cyan], '
                f'total [bold red]{len(df_no_duplicates)}[bold cyan] check-ins')

    """step 1: filter POI occurred less than 5 times"""
    poi_filtered_df = df_no_duplicates.groupby("venue_ID").filter(lambda x: len(x) >= 5)
    console.log(f'[bold cyan]after [bold red]poi filtering[bold cyan] (occurred less than [bold red]5[bold cyan] '
                f'times), total [bold red]{len(poi_filtered_df)}[bold cyan] check-ins, '
                f'current [bold red]{df_no_duplicates["user_ID"].nunique()}[bold cyan] users')

    """step 2: generate trajectory for each user"""
    filtered_user_df_list = []
    id_idx = 1
    user_grouped_df = poi_filtered_df.groupby('user_ID')
    for user_id, user_df in user_grouped_df:
        user_df = user_df.sort_values('UTC_time')
        """step 3: split sub-trajectory via 6-hour"""
        time_diff = user_df["UTC_time"].diff()
        user_df['temp_traj_id'] = (time_diff > pd.Timedelta(hours=6)).astype(int).cumsum()
        """step 4: filter duplicates poi (same poi, different time -- self-loop trip)"""
        user_df = user_df.drop_duplicates(subset=["venue_ID", "temp_traj_id"])
        user_df = user_df.reset_index(drop=True)
        """step 5: filter sub-trajectory less than 3 POIs"""
        sub_traj_poi_filtered_df = user_df[user_df['temp_traj_id'].map(user_df['temp_traj_id'].value_counts()) >= 3]
        """step 6: filter user who has less than 5 trajectories"""
        if len(sub_traj_poi_filtered_df["temp_traj_id"].unique()) >= 5:
            df_copy = sub_traj_poi_filtered_df.copy()
            df_copy['traj_id'] = pd.factorize(df_copy["temp_traj_id"])[0] + id_idx
            id_idx = df_copy['traj_id'].max() + 1
            filtered_user_df_list.append(df_copy)
        else:
            console.log(f'[bold cyan]filter user [bold red]{user_id}[bold cyan] who only has '
                        f'[bold red]{len(sub_traj_poi_filtered_df["temp_traj_id"].unique())}[bold cyan] trajectories')

    merged_df = pd.concat(filtered_user_df_list)

    console.log(f'[bold cyan]current total [bold red]{len(filtered_user_df_list)}[bold cyan] user num,'
                f'total [bold red]{len(merged_df["venue_ID"].unique())}[bold cyan] poi num,'
                f'total [bold red]{merged_df["traj_id"].max()}[bold cyan] trajectories')

    merged_df.to_csv(f'../data/{opt.dataset}.csv')

    console.log(f'file is saved as "/asset/data/{opt.dataset}.csv"')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='NYC', help='dataset')
    args = parser.parse_args()

    main(args)

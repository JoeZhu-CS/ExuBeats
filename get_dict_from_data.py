import numpy as np
import csv

def get_dict_from_data(filename: str) -> dict[str, list]:
    """
    从 CSV 文件中读取数据并返回字典。

    要求 CSV 文件包含表头，且至少包含如下字段（表头名称区分大小写）：
        - Track Name
        - Artist Name(s)
        - Popularity
        - Genres
        - Danceability
        - Energy
        - Key
        - Loudness
        - Mode
        - Speechiness
        - Acousticness
        - Instrumentalness
        - Liveness
        - Valence
        - Tempo
        - Duration (ms)

    返回的字典结构为：
        {
            track_name.lower(): [
                np.array([track_artist, playlist_genre]),  # 文本数据
                np.array([track_popularity, danceability, energy, key, loudness, mode,
                          speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms])  # 数值数据
            ],
            ...
        }
    """
    # 定义输出字段映射：输出字段名称 -> CSV 中的表头名称
    mapping = {
        'track_name': 'Track Name',
        'track_artist': 'Artist Name(s)',
        'track_popularity': 'Popularity',
        'playlist_genre': 'Genres',
        'danceability': 'Danceability',
        'energy': 'Energy',
        'key': 'Key',
        'loudness': 'Loudness',
        'mode': 'Mode',
        'speechiness': 'Speechiness',
        'acousticness': 'Acousticness',
        'instrumentalness': 'Instrumentalness',
        'liveness': 'Liveness',
        'valence': 'Valence',
        'tempo': 'Tempo',
        'duration_ms': 'Duration (ms)'
    }

    # 文本数据只需要提取 track_artist 与 playlist_genre 两个字段
    text_fields = ['track_artist', 'playlist_genre']
    # 数值数据按顺序提取：Popularity, Danceability, Energy, Key, Loudness, Mode,
    # Speechiness, Acousticness, Instrumentalness, Liveness, Valence, Tempo, Duration (ms)
    numeric_fields = ['track_popularity', 'danceability', 'energy', 'key', 'loudness', 'mode',
                      'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']

    final_dict = {}
    with open(filename, encoding="utf8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            # 根据 CSV 表头提取歌曲名称，作为字典的 key（统一转为小写）
            track_name = row.get(mapping['track_name'], '').strip()
            if not track_name:
                continue

            # 提取文本数据（艺术家和流派），原样保留
            text_data = [row.get(mapping[field], '').strip() for field in text_fields]

            # 提取数值数据，并转换为 float 类型
            try:
                numeric_data = [float(row.get(mapping[field], 0)) for field in numeric_fields]
            except ValueError:
                # 如果转换失败则跳过该行
                continue

            final_dict[track_name.lower()] = [np.array(text_data), np.array(numeric_data)]
    return final_dict

from MusicRecommender import MusicRecommender
import os
from custom_weight_selector import customize_weights
import plotly.io as pio
pio.renderers.default = "browser"


if __name__ == "__main__":
    import doctest
    doctest.testmod()

    # 设置数据文件路径，使用新的 CSV 文件
    data_file = 'arknights_ep.csv'
    if not os.path.exists(data_file):
        print(f"错误：未找到数据文件 '{data_file}'。")
        print("请确保数据文件在当前目录中。")
        exit(1)

    # 创建推荐系统对象
    recommender = MusicRecommender(data_file)

    # 显示欢迎信息
    print("\n===== SoundSage音乐推荐系统 =====")
    print("本系统帮助你根据喜欢的歌曲推荐相似的音乐。")
    print("你可以选择基本推荐模式或自定义偏好模式。")

    # 定义查找歌曲函数
    def find_song(search_term):
        """
        根据输入的歌曲名称查找合适的歌曲
        """
        normalized_search = search_term.lower().strip()

        # 直接匹配
        if normalized_search in recommender.song_dict:
            return normalized_search

        # 去除多余空格后匹配
        compressed_search = ''.join(normalized_search.split())
        for song_name in recommender.song_dict.keys():
            compressed_song = ''.join(song_name.split())
            if compressed_search == compressed_song:
                return song_name

        # 尝试部分匹配
        potential_matches = []
        for song_name in recommender.song_dict.keys():
            if normalized_search in song_name:
                potential_matches.append(song_name)

        if potential_matches:
            if len(potential_matches) > 1:
                print("\n找到多首匹配歌曲，请选择：")
                for i, match in enumerate(potential_matches[:10], 1):
                    artist = recommender.song_dict[match][0][0]
                    print(f"{i}. {match.title()} - {artist}")
                while True:
                    try:
                        selection = int(input("\n请输入对应数字（或输入0重新搜索）："))
                        if selection == 0:
                            return None
                        if 1 <= selection <= len(potential_matches[:10]):
                            return potential_matches[selection - 1]
                        print("选择无效，请重试。")
                    except ValueError:
                        print("请输入数字。")
            else:
                return potential_matches[0]

        return None

    # 主循环
    while True:
        print("\n请输入歌曲名称以获取推荐：")
        print("例如：'Operation Pyrite'、'INFECTED'、'Speed of Light' 等")
        song_input = input("歌曲名称（或输入 'exit' 退出）：")

        if song_input.lower() == 'exit':
            print("\n感谢使用SoundSage音乐推荐系统！")
            break

        song_name = find_song(song_input)

        if not song_name:
            print(f"抱歉，数据集中未找到 '{song_input}'。")
            print("请尝试其他歌曲名称。")
            continue

        song_title = song_name
        song_data = recommender.song_dict[song_name]
        artist = song_data[0][0]
        print(f"\n找到：'{song_name.title()}'  艺术家：{artist}")

        try:
            # 显示歌曲特征
            print("\n--- 歌曲特征分析 ---")
            print("是否显示该歌曲的音频特征？")
            if input("显示音频特征？（yes/no）：").lower().startswith('y'):
                recommender.visualize_song_features(song_title)
                print("音频特征图已显示。")

            # 推荐模式选择
            print("\n--- 推荐模式选择 ---")
            print("推荐模式有两种：")
            print("1. 基本模式：使用默认设置查找相似歌曲")
            print("2. 高级模式：可自定义偏好（如艺术家、流派、节奏等）")
            while True:
                mode_input = input("请选择模式（basic/advanced）：").strip().lower()
                if mode_input.startswith('a'):
                    use_preferences = True
                    break
                elif mode_input.startswith('b'):
                    use_preferences = False
                    break
                else:
                    print("选择无效，请输入 'basic' 或 'advanced'。")

            preferences = None
            if use_preferences:
                num_input = input("请输入推荐歌曲数量（默认10）：").strip()
                try:
                    num_recs = int(num_input) if num_input else 10
                except ValueError:
                    print("输入无效，默认数量设为10。")
                    num_recs = 10

                print("\n--- 设置推荐偏好 ---")
                preferences = recommender.get_setting_preferences(song_title)

                print("\n--- 特征权重自定义 ---")
                print("推荐系统使用不同的音频特征权重进行相似度计算。")
                if input("是否查看默认特征权重？（yes/no）：").lower().startswith('y'):
                    weights = {
                        'genre': 0.15,
                        'artist': 0.05,
                        'popularity': 0.05,
                        'danceability': 0.10,
                        'energy': 0.10,
                        'key': 0.02,
                        'loudness': 0.05,
                        'mode': 0.02,
                        'speechiness': 0.08,
                        'acousticness': 0.08,
                        'instrumentalness': 0.08,
                        'liveness': 0.05,
                        'valence': 0.10,
                        'tempo': 0.05,
                        'duration_ms': 0.02
                    }
                    print("正在显示特征权重图...")
                    recommender.visualize_feature_weights(weights)
                    print("特征权重图已显示。")

                print("\n--- 自定义特征权重推荐 ---")
                custom_weights = customize_weights()
                if input("是否查看自定义后的特征权重图？（yes/no）：").lower().startswith('y'):
                    print("正在显示特征权重图...")
                    feature_names = [
                        'popularity', 'danceability', 'energy', 'key', 'loudness',
                        'mode', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo',
                        'duration'
                    ]
                    weights_dict = dict(zip(feature_names, custom_weights))
                    recommender.visualize_feature_weights(weights_dict)
                    print("特征权重图已显示。")

                try:
                    rec_songs = recommender.generate_similarity_list(song_title, n=num_recs, preferences=preferences,
                                                                     weights=custom_weights)
                    print("\n使用自定义权重推荐的相似歌曲：")
                    for idx, song in enumerate(rec_songs, 1):
                        print(f"{idx}. {song}")
                except ValueError as e:
                    print(f"错误：{e}")

                if preferences is None:
                    print("\n请先设置过滤偏好：")
                    preferences = recommender.get_setting_preferences(song_title)
                else:
                    print("使用之前设置的偏好与自定义权重...")

            else:
                print("\n--- 基本推荐 ---")
                print("正在使用默认参数查找相似歌曲...")
                similar_songs = recommender.find_similar_songs(song_title, 5)
                print(f"\n为 '{song_title.title()}' 推荐的 {len(similar_songs)} 首相似歌曲：")
                for i, song in enumerate(similar_songs, 1):
                    similar_artist = recommender.song_dict[song][0][0]
                    similar_genre = recommender.song_dict[song][0][1]
                    print(f"{i}. {song.title()} - {similar_artist} （流派：{similar_genre}）")

            # 网络可视化
            print("\n--- 歌曲网络可视化 ---")
            print("是否需要可视化歌曲网络？")
            if input("显示网络可视化？（yes/no）：").lower().startswith('y'):
                print("正在创建歌曲网络图...")
                recommender.visualize_similar_songs([song_title], max_similar=8, threshold=0.6,
                                                    use_preferences=preferences is not None,
                                                    existing_preferences=preferences)
                print("网络可视化已显示。")

            # 多歌曲比较
            print("\n--- 多歌曲比较 ---")
            print("是否添加另一首歌曲进行比较？")
            if input("添加另一首歌曲？（yes/no）：").lower().startswith('y'):
                second_song_input = input("请输入另一首歌曲名称：")
                second_song = find_song(second_song_input)
                if second_song:
                    print(f"正在为 '{song_title.title()}' 和 '{second_song.title()}' 创建网络图...")
                    multi_use_prefs = input("是否在比较中使用之前设置的偏好？（yes/no）：").lower().startswith('y')
                    recommender.visualize_similar_songs([song_title, second_song], max_similar=5,
                                                        use_preferences=multi_use_prefs,
                                                        existing_preferences=preferences if multi_use_prefs else None)
                    print("多歌曲网络可视化已显示。")
                else:
                    print(f"抱歉，数据集中未找到 '{second_song_input}'。")

            print("\n是否继续尝试其他歌曲？")
            if not input("继续？（yes/no）：").lower().startswith('y'):
                print("\n感谢使用SoundSage音乐推荐系统！")
                break

        except Exception as e:
            print(f"发生错误：{e}")
            print("请重试。")

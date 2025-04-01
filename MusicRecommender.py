import networkx as nx
from plotly.graph_objs import Scatterpolar, Scatter, Figure, Bar
from typing import Dict, List, Optional
from get_dict_from_data import get_dict_from_data


# 颜色方案及相关常量
COLOUR_SCHEME = [
    '#2E91E5', '#E15F99', '#1CA71C', '#FB0D0D', '#DA16FF', '#222A2A', '#B68100',
    '#750D86', '#EB663B', '#511CFB', '#00A08B', '#FB00D1', '#FC0080', '#B2828D',
    '#6C7C32', '#778AAE', '#862A16', '#A777F1', '#620042', '#1616A7', '#DA60CA',
    '#6C4516', '#0D2A63', '#AF0038'
]

LINE_COLOUR = 'rgb(210,210,210)'
VERTEX_BORDER_COLOUR = 'rgb(50, 50, 50)'
SONG_COLOUR = 'rgb(89, 205, 105)'
INPUT_SONG_COLOUR = 'rgb(105, 89, 205)'


class MusicRecommender:
    """
    音乐推荐系统，用于根据输入歌曲推荐相似的歌曲，并提供多种可视化展示。
    该系统依赖 get_dict_from_data 函数加载 CSV 数据，
    CSV 文件要求包含如下字段（表头名称区分大小写）：
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

    数据将转换为内部统一格式：
      - 文本数据：[track_artist, playlist_genre]
      - 数值数据：[track_popularity, danceability, energy, key, loudness, mode,
                     speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms]
    """

    def __init__(self, song_data_path: str):
        """
        初始化推荐系统对象

        参数:
            song_data_path: CSV 数据文件路径（文件需满足上述字段要求）
        """
        self.song_dict = get_dict_from_data(song_data_path)

    def get_setting_preferences(self, song_name: str) -> tuple[bool, bool, bool, bool, float, float, float, float]:
        """
        获取用户偏好设置，根据参考歌曲收集过滤条件

        参数:
            song_name: 参考歌曲名称

        返回:
            (same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration)
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"数据集中未找到歌曲：'{song_name}'")

        song = self.song_dict[song_name.lower()]

        # 限制同一艺术家
        answer = input("是否只推荐相同艺术家的歌曲？（输入 Yes 或 No）：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否只推荐相同艺术家的歌曲？（输入 Yes 或 No）：").strip().lower()
        same_artist = answer in ["yes", "y"]

        # 限制同一流派
        answer = input("是否只推荐相同流派的歌曲？（输入 Yes 或 No）：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否只推荐相同流派的歌曲？（输入 Yes 或 No）：").strip().lower()
        same_genre = answer in ["yes", "y"]

        # 限制同一调性
        answer = input("是否只推荐相同调性的歌曲？（输入 Yes 或 No）：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否只推荐相同调性的歌曲？（输入 Yes 或 No）：").strip().lower()
        same_key = answer in ["yes", "y"]

        # 限制同一模式（大调/小调）
        answer = input("是否只推荐相同模式（大调/小调）的歌曲？（输入 Yes 或 No）：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否只推荐相同模式（大调/小调）的歌曲？（输入 Yes 或 No）：").strip().lower()
        same_mode = answer in ["yes", "y"]

        # 限制节奏
        answer = input(
            f"是否基于节奏限制推荐？参考歌曲 \"{song_name}\" 的节奏为 {song[1][11]} bpm，请输入 Yes 或 No：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否基于节奏限制推荐？请输入 Yes 或 No：").strip().lower()
        if answer in ["yes", "y"]:
            lower_bound_tempo = float(input("请输入节奏下限（bpm，整数）：").strip())
            upper_bound_tempo = float(input("请输入节奏上限（bpm，整数）：").strip())
        else:
            lower_bound_tempo = 0
            upper_bound_tempo = float('inf')

        # 限制时长
        answer = input(
            f"是否基于时长限制推荐？参考歌曲 \"{song_name}\" 的时长为 {int(song[1][12] / 1000)} 秒，请输入 Yes 或 No：").strip().lower()
        while answer not in ["yes", "no", "y", "n"]:
            print("答案无效，请重试。")
            answer = input("是否基于时长限制推荐？请输入 Yes 或 No：").strip().lower()
        if answer in ["yes", "y"]:
            lower_bound_duration = float(input("请输入时长下限（秒）：").strip()) * 1000
            upper_bound_duration = float(input("请输入时长上限（秒）：").strip()) * 1000
        else:
            lower_bound_duration = 0
            upper_bound_duration = float('inf')

        return same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration

    def generate_similarity_list(self, song_name: str, n: int = 10,
                                 preferences: Optional[
                                     tuple[bool, bool, bool, bool, float, float, float, float]] = None,
                                 weights: Optional[List[float]] = None) -> List[str]:
        """
        根据用户偏好和特征权重生成相似歌曲列表

        参数:
            song_name: 参考歌曲名称
            n: 返回相似歌曲数量
            preferences: 用户过滤偏好
            weights: 特征权重列表
                     顺序为 [track_popularity, danceability, energy, key, loudness, mode,
                              speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration_ms]

        返回:
            相似歌曲名称列表
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"数据集中未找到歌曲：'{song_name}'")

        if weights is None:
            weights = [0.05, 0.12, 0.12, 0.05, 0.05, 0.05, 0.05, 0.08, 0.08, 0.05, 0.15, 0.10, 0.05]

        original_song = self.song_dict[song_name.lower()]
        original_song_data = original_song[1]

        same_artist = same_genre = same_key = same_mode = False
        lower_bound_tempo = lower_bound_duration = float('-inf')
        upper_bound_tempo = upper_bound_duration = float('inf')

        og_artist = og_genre = None
        og_key = og_mode = None

        if preferences:
            same_artist, same_genre, same_key, same_mode, lower_bound_tempo, upper_bound_tempo, lower_bound_duration, upper_bound_duration = preferences
            og_artist = original_song[0][0]
            og_genre = original_song[0][1]
            og_key = original_song_data[3]
            og_mode = original_song_data[5]

        similarities = {}
        for other_song_name, other_song in self.song_dict.items():
            if other_song_name == song_name.lower():
                continue

            if preferences:
                if ((same_artist and og_artist != other_song[0][0]) or
                        (same_genre and og_genre != other_song[0][1]) or
                        (same_key and og_key != other_song[1][3]) or
                        (same_mode and og_mode != other_song[1][5]) or
                        (other_song[1][11] < lower_bound_tempo) or
                        (other_song[1][11] > upper_bound_tempo) or
                        (other_song[1][12] < lower_bound_duration) or
                        (other_song[1][12] > upper_bound_duration)):
                    continue

            distance = 0
            other_features = other_song[1]
            for i, (feat1, feat2) in enumerate(zip(original_song_data, other_features)):
                if i == 3:  # key
                    normalized_diff = abs(feat1 - feat2) / 11
                    distance += weights[i] * normalized_diff ** 2
                elif i == 4:  # loudness
                    normalized_diff = abs((feat1 + 60) / 60 - (feat2 + 60) / 60)
                    distance += weights[i] * normalized_diff ** 2
                elif i == 11:  # tempo，不做实际归一化处理
                    normalized_diff = abs(feat1 - feat2) / float('inf')
                    distance += weights[i] * normalized_diff ** 2
                elif i == 12:  # duration_ms
                    normalized_diff = abs(feat1 - feat2) / (5 * 60 * 1000)
                    distance += weights[i] * normalized_diff ** 2
                else:
                    distance += weights[i] * (feat1 - feat2) ** 2
            similarity = 1 / (1 + distance ** 0.5)
            similarities[other_song_name] = similarity

        sorted_songs = sorted(similarities.items(), key=lambda x: x[1], reverse=True)
        return [song for song, _ in sorted_songs[:n]]

    def find_similar_songs(self, song_name: str, n: int = 10, use_preferences: bool = False) -> List[str]:
        """
        查找与指定歌曲相似的歌曲

        参数:
            song_name: 参考歌曲名称
            n: 返回相似歌曲数量
            use_preferences: 是否使用用户偏好进行过滤

        返回:
            相似歌曲名称列表
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"数据集中未找到歌曲：'{song_name}'")

        if use_preferences:
            print("开始收集用于歌曲推荐的偏好设置。")
            preferences = self.get_setting_preferences(song_name)
            return self.generate_similarity_list(song_name, n, preferences)
        return self.generate_similarity_list(song_name, n)

    def visualize_similar_songs(self, song_names: List[str], max_similar: int = 5, threshold: float = 0.7,
                                output_file: str = '', use_preferences: bool = False,
                                existing_preferences: Optional[tuple] = None) -> None:
        """
        可视化与输入歌曲相似的歌曲网络

        参数:
            song_names: 输入歌曲名称列表
            max_similar: 每首歌曲显示的最多相似歌曲数
            threshold: 包含边的最小相似度
            output_file: 若为空，则在浏览器中显示，否则保存到指定文件
            use_preferences: 是否使用用户偏好进行过滤
            existing_preferences: 已收集的用户偏好（可复用）
        """
        valid_songs = []
        for song in song_names:
            if song.lower() in self.song_dict:
                valid_songs.append(song)
            else:
                print(f"警告：数据集中未找到歌曲 '{song}'")
        if not valid_songs:
            raise ValueError("未提供有效歌曲。")

        all_similar_songs = {}
        preferences_to_use = None

        for song in valid_songs:
            if use_preferences:
                if existing_preferences:
                    preferences_to_use = existing_preferences
                    print(f"使用之前设置的偏好为 '{song}' 推荐歌曲。")
                else:
                    print(f"开始收集 '{song}' 的推荐偏好。")
                    preferences_to_use = self.get_setting_preferences(song)
            similar_songs = self.generate_similarity_list(song, max_similar * 3, preferences_to_use)
            similarity_scores = {}
            for similar_song in similar_songs:
                pos = similar_songs.index(similar_song)
                similarity_scores[similar_song] = 1 - (pos / (len(similar_songs) + 1))
            all_similar_songs[song.lower()] = similarity_scores

        graph = self._create_song_graph(valid_songs, all_similar_songs, threshold, max_similar)
        self._visualize_song_graph(graph, output_file=output_file)

    def visualize_song_features(self, song_name: str, output_file: str = '') -> None:
        """
        可视化指定歌曲的音频特征

        参数:
            song_name: 要可视化的歌曲名称
            output_file: 若为空，则在浏览器中显示，否则保存到指定文件
        """
        if song_name.lower() not in self.song_dict:
            raise ValueError(f"数据集中未找到歌曲：'{song_name}'")

        song_data = self.song_dict[song_name.lower()]
        num_data = song_data[1]

        features = [
            'Danceability', 'Energy', 'Key', 'Loudness', 'Mode',
            'Speechiness', 'Acousticness', 'Instrumentalness',
            'Liveness', 'Valence', 'Tempo'
        ]
        normalized_values = [
            num_data[1],  # Danceability
            num_data[2],  # Energy
            num_data[3] / 11,  # Key (归一化到0-1)
            (num_data[4] + 60) / 60,  # Loudness (归一化到0-1)
            num_data[5],  # Mode
            num_data[6],  # Speechiness
            num_data[7],  # Acousticness
            num_data[8],  # Instrumentalness
            num_data[9],  # Liveness
            num_data[10],  # Valence
            num_data[11] / 250  # Tempo (归一化到0-1，假设最大值250)
        ]
        # 闭合雷达图
        features.append(features[0])
        normalized_values.append(normalized_values[0])

        fig = Figure()
        fig.add_trace(Scatterpolar(
            r=normalized_values,
            theta=features,
            fill='toself',
            name=song_name
        ))
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )
            ),
            title=f"音频特征分析：{song_name} - {song_data[0][0]}",
            title_x=0.5
        )
        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)

    def _create_song_graph(self, input_songs: List[str], similarity_scores: Dict[str, Dict[str, float]],
                           threshold: float = 0.7, max_connections: int = 5) -> nx.Graph:
        """
        根据相似度数据构建歌曲网络图

        参数:
            input_songs: 输入歌曲列表
            similarity_scores: 每首歌曲对应的相似度字典
            threshold: 包含边的最小相似度
            max_connections: 每首歌曲最多显示的相似歌曲数

        返回:
            NetworkX 图对象
        """
        G = nx.Graph()

        for song in input_songs:
            if song.lower() in self.song_dict:
                G.add_node(song.lower(), kind='input_song',
                           artist=self.song_dict[song.lower()][0][0],
                           genre=self.song_dict[song.lower()][0][1])
        for input_song, scores in similarity_scores.items():
            sorted_songs = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            for similar_song, score in sorted_songs[:max_connections]:
                if score >= threshold and similar_song in self.song_dict:
                    if similar_song not in G:
                        G.add_node(similar_song, kind='song',
                                   artist=self.song_dict[similar_song][0][0],
                                   genre=self.song_dict[similar_song][0][1])
                    G.add_edge(input_song, similar_song, weight=score)
        return G

    def _visualize_song_graph(self, graph: nx.Graph, layout: str = 'spring_layout', max_vertices: int = 100,
                              output_file: str = '') -> None:
        """
        可视化歌曲相似性网络

        参数:
            graph: 要可视化的 NetworkX 图对象
            layout: 布局方式（默认 spring_layout）
            max_vertices: 最大显示顶点数
            output_file: 若为空，则在浏览器中显示，否则保存到指定文件
        """
        if len(graph) > max_vertices:
            graph = graph.subgraph(list(graph.nodes)[:max_vertices])

        pos = getattr(nx, layout)(graph)
        x_values = [pos[k][0] for k in graph.nodes]
        y_values = [pos[k][1] for k in graph.nodes]
        labels = [f"{k} - {graph.nodes[k]['artist']}" for k in graph.nodes]
        kinds = [graph.nodes[k]['kind'] for k in graph.nodes]
        colours = [INPUT_SONG_COLOUR if kind == 'input_song' else SONG_COLOUR for kind in kinds]

        x_edges = []
        y_edges = []
        for edge in graph.edges:
            x_edges += [pos[edge[0]][0], pos[edge[1]][0], None]
            y_edges += [pos[edge[0]][1], pos[edge[1]][1], None]
        edge_trace = Scatter(
            x=x_edges,
            y=y_edges,
            mode='lines',
            name='相似度',
            line=dict(color=LINE_COLOUR, width=1),
            hoverinfo='none'
        )
        node_trace = Scatter(
            x=x_values,
            y=y_values,
            mode='markers',
            name='歌曲',
            marker=dict(
                symbol='circle',
                size=10,
                color=colours,
                line=dict(color=VERTEX_BORDER_COLOUR, width=0.5)
            ),
            text=labels,
            hovertemplate='%{text}<br>流派: %{customdata[0]}',
            customdata=[[graph.nodes[k]['genre']] for k in graph.nodes],
            hoverlabel={'namelength': 0}
        )
        data = [edge_trace, node_trace]
        fig = Figure(data=data)
        fig.update_layout({
            'showlegend': False,
            'title': '歌曲相似性网络',
            'title_x': 0.5,
            'hovermode': 'closest'
        })
        fig.update_xaxes(showgrid=False, zeroline=False, visible=False)
        fig.update_yaxes(showgrid=False, zeroline=False, visible=False)
        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)

    def visualize_feature_weights(self, weights: Dict[str, float], output_file: str = '') -> None:
        """
        可视化相似度计算中各特征的权重

        参数:
            weights: 字典，键为特征名称，值为对应权重
            output_file: 若为空，则在浏览器中显示，否则保存到指定文件
        """
        sorted_features = sorted(weights.items(), key=lambda x: x[1], reverse=True)
        feature_names = [f[0] for f in sorted_features]
        feature_weights = [f[1] for f in sorted_features]

        fig = Figure(data=[
            Bar(
                x=feature_names,
                y=feature_weights,
                marker=dict(color='rgb(55, 83, 109)')
            )
        ])
        fig.update_layout(
            title='歌曲推荐中特征的重要性',
            title_x=0.5,
            xaxis=dict(
                title='特征',
                tickangle=45
            ),
            yaxis=dict(
                title='权重',
                range=[0, max(feature_weights) * 1.1]
            ),
            margin=dict(b=100)
        )
        if output_file == '':
            fig.show()
        else:
            fig.write_image(output_file)

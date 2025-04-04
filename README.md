# SoundSage
SoundSage 是一款基于 CSV 数据的音乐推荐系统。它利用用户自定义的过滤偏好和加权欧氏距离，为您推荐与所选歌曲相似的音乐，并通过 Plotly 和 NetworkX 提供音频特征、特征权重和歌曲相似性网络的直观可视化。

# 特点
CSV 数据读取：自动根据表头提取并重排所需字段，支持额外数据列。

个性化推荐：通过命令行交互输入歌曲，设置过滤条件（如艺术家、流派、调性、模式、节奏、时长）以及自定义特征权重。

相似度计算：采用加权欧氏距离算法计算歌曲间的相似性。

可视化展示：支持音频特征雷达图、特征权重条形图和歌曲相似性网络图。

交互式命令行界面：流程清晰、提示友好，易于使用。

# 使用方法
运行 main.py

按照提示输入歌曲名称，系统将尝试匹配并展示对应歌曲的信息。

根据提示选择推荐模式：

  基本模式：使用默认参数查找相似歌曲。

  高级模式：设置过滤偏好（如同一艺术家、同一流派、调性、模式、节奏和时长范围）后，再自定义特征权重。

系统会生成推荐列表，并通过图形界面展示音频特征、特征权重以及歌曲相似性网络图。

根据提示，您可以进一步进行多歌曲比较等操作。

# 项目结构
get_dict_from_data.py
读取 CSV 数据文件，根据表头提取所需字段，生成内部统一格式的数据字典。

MusicRecommender.py
包含核心推荐算法、用户偏好设置以及可视化功能（雷达图、条形图、网络图）。

custom_weight_selector.py
提供自定义特征权重的交互式选择界面。

main.py
程序入口，控制整体交互流程与用户操作。

arknights_ep.csv
示例 CSV 数据文件，需包含预期的表头字段。

# 贡献
欢迎提交问题和建议！如果您想为 SoundSage 贡献代码，请通过 Issue 联系我。

SoundSage 致力于为音乐爱好者提供简单、高效、个性化的音乐推荐体验。欢迎体验并反馈您的使用感受！

def customize_weights(default_weights=None, max_selection=5, high_value=0.15, low_value=0.05):
    """
    允许用户选择在歌曲推荐中优先考虑的特征权重。

    如果用户直接回车，则返回默认权重；如果输入指定特征序号（逗号分隔），
    则这些特征赋予高优先级权重，其它特征赋予低优先级权重，最终将权重归一化。

    参数:
        default_weights: 默认权重列表。如果为 None，则使用预设默认值。
        max_selection: 允许强调的特征数量上限（最多选择的特征个数）。
        high_value: 被强调特征的初始权重值。
        low_value: 未被强调特征的初始权重值。

    返回:
        归一化后的权重列表。
    """
    # 定义各特征名称，顺序需与 MusicRecommender 中数值数据的顺序对应
    param_names = [
        "流行度", "舞动性", "能量", "调性", "响度",
        "模式", "语速", "原声性", "乐器性",
        "现场感", "情绪", "节奏", "时长"
    ]

    if default_weights is None:
        default_weights = [0.05, 0.15, 0.15, 0.05, 0.05, 0.05, 0.05, 0.1, 0.1, 0.05, 0.1, 0.05, 0.05]

    print("\n默认特征权重：")
    for i, (name, weight) in enumerate(zip(param_names, default_weights), start=1):
        print(f"{i}. {name}: {weight}")

    indices_input = input(
        f"\n请输入你希望在推荐过程中给予更高优先级的特征序号（用逗号分隔，最多 {max_selection} 个）。\n"
        "直接回车则使用默认权重： "
    ).strip()

    if not indices_input:
        return default_weights

    try:
        indices = [int(x.strip()) for x in indices_input.split(",") if x.strip()]
    except ValueError:
        print("输入无效，使用默认权重。")
        return default_weights

    valid_indices = []
    for idx in indices:
        if 1 <= idx <= len(param_names) and idx not in valid_indices:
            valid_indices.append(idx)
    if len(valid_indices) > max_selection:
        print(f"最多只能选择 {max_selection} 个特征，现取前 {max_selection} 个有效选择。")
        valid_indices = valid_indices[:max_selection]

    new_raw_weights = []
    for i in range(len(param_names)):
        if (i + 1) in valid_indices:
            new_raw_weights.append(high_value)
        else:
            new_raw_weights.append(low_value)

    total = sum(new_raw_weights)
    normalized_weights = [w / total for w in new_raw_weights]

    print("\n自定义归一化后的特征权重：")
    for i, (name, weight) in enumerate(zip(param_names, normalized_weights), start=1):
        print(f"{i}. {name}: {weight:.4f}")

    return normalized_weights


if __name__ == "__main__":
    # 简单测试：直接运行此模块时，可测试自定义权重函数
    final_weights = customize_weights()
    print("最终归一化后的权重：", final_weights)

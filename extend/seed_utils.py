def load_seeds(file_path):
    """
    从txt文件中加载种子数据并转换为整数列表
    
    Args:
        file_path (str): txt文件路径
        
    Returns:
        list: 包含所有种子的整数列表
    """
    seed_list = []
    
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # 去除行首尾空白字符
                line = line.strip()
                if not line:  # 跳过空行
                    continue
                
                # 按空格分割字符串并转换为整数
                numbers = line.split()
                for num_str in numbers:
                    try:
                        seed = int(num_str)
                        seed_list.append(seed)
                    except ValueError:
                        print(f"警告: 无法将 '{num_str}' 转换为整数，已跳过")
    
    except FileNotFoundError:
        print(f"错误: 文件 '{file_path}' 未找到")
        return []
    except Exception as e:
        print(f"读取文件时出错: {e}")
        return []
    
    return seed_list

def split_seeds(seed_list, gpu_num, verbose=False):
    """
    拆分 seed_list 为 gpu_num 份
    
    Args:
        seed_list: 原始列表
        gpu_num: 要拆分的份数
    
    Returns:
        dict: 包含拆分结果和统计信息的字典
    """

    total_length = len(seed_list)
    usable_length = total_length // gpu_num * gpu_num
    dropped_count = total_length - usable_length
    
    truncated_list = seed_list[:usable_length]
    
    # 创建 n 个子列表
    subsets = [[] for _ in range(gpu_num)]
    
    # 分配元素到各个子集
    for i, item in enumerate(truncated_list):
        subset_index = i % gpu_num
        subsets[subset_index].append(item)
    
    if verbose:
        print(f"种子总数{total_length}")
        print(f"使用前{usable_length}个，丢弃{dropped_count}个")
    return subsets

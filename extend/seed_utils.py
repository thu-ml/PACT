def load_seeds(file_path):
    seed_list = []
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            for line in file:
                line = line.strip()
                if not line:
                    continue
                
                numbers = line.split()
                for num_str in numbers:
                    try:
                        seed = int(num_str)
                        seed_list.append(seed)
                    except ValueError:
                        print(f"WARNING: cannot convert '{num_str}' to int, continue")
    
    except FileNotFoundError:
        print(f"Error: file '{file_path}' not found")
        return []
    except Exception as e:
        print(f"Error: reading file: {e}")
        return []
    
    return seed_list

def split_seeds(seed_list, gpu_num, verbose=False):
    total_length = len(seed_list)
    usable_length = total_length // gpu_num * gpu_num
    dropped_count = total_length - usable_length
    
    truncated_list = seed_list[:usable_length]
    
    subsets = [[] for _ in range(gpu_num)]
    
    for i, item in enumerate(truncated_list):
        subset_index = i % gpu_num
        subsets[subset_index].append(item)
    
    if verbose:
        print(f"total length{total_length}")
        print(f"use first {usable_length}, drop {dropped_count}")
    return subsets

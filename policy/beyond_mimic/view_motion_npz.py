import numpy as np
import os

def view_npz_data(file_path='motion.npz', num_rows=5):
    """
    加载并打印 .npz 文件中每个数组的名称、形状和前几行数据。

    Args:
        file_path (str): .npz 文件的路径。
        num_rows (int): 要打印的数据行数。
    """
    if not os.path.exists(file_path):
        print(f"错误：文件未找到在 {file_path}")
        return

    try:
        # 使用 np.load 加载 .npz 文件
        # allow_pickle=True 在加载包含 Python 对象（如字典或列表）的数组时可能需要，
        # 如果你确定文件只包含标准数组，可以省略或设为 False。
        data = np.load(file_path, allow_pickle=True)

        print(f"成功加载文件：{file_path}\n")

        # .npz 文件是一个类似字典的对象，它的键是保存数组时的名称
        array_names = data.files
        print(f"文件包含以下数组（键）：{array_names}\n" + "="*50)

        for name in array_names:
            array = data[name]
            
            print(f"--- 数组名称: **{name}** ---")
            print(f"形状 (Shape): {array.shape}")
            print(f"数据类型 (Dtype): {array.dtype}")

            # 检查数组是否有足够的数据行可以打印
            if array.ndim >= 1 and array.shape[0] > 0:
                # 打印前 num_rows 行
                rows_to_print = min(num_rows, array.shape[0])
                print(f"前 {rows_to_print} 行数据:")
                
                # 使用切片来获取前几行
                print(array[:num_rows])
            elif array.ndim == 0:
                # 针对标量或单个元素的数组
                print("这是一个标量或单个元素数组 (ndim=0)，其值为:")
                print(array)
            else:
                print("数组为空或维度不适合查看前几行数据。")
            
            print("-" * 50)

    except Exception as e:
        print(f"加载或处理文件时发生错误: {e}")

# --- 使用示例 ---
# 假设你的文件名为 motion.npz，并且它在当前目录下
# file_name = '/home/ethanlee/project/BeyondMimic/beyond_mimic_tracking/artifacts/s3_d1s2_trim:v0/motion.npz'
file_name = '/home/ethanlee/project/BeyondMimic/beyond_mimic_tracking/artifacts/dance1_subject2:v0/motion.npz'
rows_to_display = 5 

view_npz_data(file_path=file_name, num_rows=rows_to_display)
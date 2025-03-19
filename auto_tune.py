import subprocess
import re
import random
import os
import time
from concurrent.futures import ProcessPoolExecutor
import numpy as np
import concurrent.futures
import json

# 定义需要调整的参数及其范围
PARAMS = {
    'JUMP_VISCOSITY': (0.1, 1.5),
    'READ_ROUND_TIME': (2, 100),
    'PRE_DISTRIBUTION_TIME': (2, 100),
    'TEST_DENSITY_LEN': (2, 2000),
    'DISK_MIN_PASS': (2, 15),
    'NUM_PIECE_QUEUE': (5, 105)
}

# 定义正则表达式模式
REGEX_PATTERNS = {
    'JUMP_VISCOSITY': r'(const\s+double\s+JUMP_VISCOSITY\s*=\s*)([0-9.]+)',
    'READ_ROUND_TIME': r'(const\s+int\s+READ_ROUND_TIME\s*=\s*)([0-9]+)',
    'PRE_DISTRIBUTION_TIME': r'(const\s+int\s+PRE_DISTRIBUTION_TIME\s*=\s*)([0-9]+)',
    'TEST_DENSITY_LEN': r'(const\s+int\s+TEST_DENSITY_LEN\s*=\s*)([0-9]+)',
    'DISK_MIN_PASS': r'(int\s+DISK_MIN_PASS\s*=\s*)([0-9]+)',
    'NUM_PIECE_QUEUE': r'(const\s+int\s+NUM_PIECE_QUEUE\s*=\s*)([0-9]+)'
}

# 全局变量
data_files = ['sample.in']

def modify_parameters(params):
    """修改code_craft.cpp中的参数"""
    try:
        # 如果code_craft.cpp不存在或为空，从main.cpp复制内容
        if not os.path.exists('code_craft.cpp') or os.path.getsize('code_craft.cpp') == 0:
            if os.path.exists('main.cpp'):
                print("从main.cpp复制内容到code_craft.cpp")
                with open('main.cpp', 'r', encoding='utf-8') as f:
                    content = f.read()
                with open('code_craft.cpp', 'w', encoding='utf-8') as f:
                    f.write(content)
            else:
                print("错误：main.cpp 文件不存在")
                return False
            
        # 读取文件内容
        with open('code_craft.cpp', 'r', encoding='utf-8') as f:
            content = f.read()
            
        if not content:
            print("错误：code_craft.cpp 文件为空")
            return False
            
        # 保存原始内容的长度
        original_length = len(content)
        
        # 对每个参数进行修改
        for param, value in params.items():
            pattern = REGEX_PATTERNS[param]
            # 检查是否找到匹配
            if not re.search(pattern, content):
                print(f"警告：未找到参数 {param} 的匹配")
                continue
                
            # 进行替换
            if param == 'JUMP_VISCOSITY':
                content = re.sub(pattern, lambda m: f'{m.group(1)}{value:.1f}', content)
            else:
                content = re.sub(pattern, lambda m: f'{m.group(1)}{int(value)}', content)
        
        # 检查修改后的内容是否为空
        if not content:
            print("错误：修改后的内容为空")
            return False
            
        # 检查内容长度是否发生显著变化
        if len(content) < original_length * 0.5:
            print(f"警告：修改后的内容长度显著减少（原始：{original_length}，现在：{len(content)}）")
            
        # 写入修改后的内容
        with open('code_craft.cpp', 'w', encoding='utf-8') as f:
            f.write(content)
            
        print("参数修改成功")
        return True
        
    except Exception as e:
        print(f"修改参数时发生错误：{e}")
        return False

def compile_code():
    """编译代码"""
    try:
        # 等待一小段时间，确保文件已经写入
        time.sleep(1)
        # 使用g++编译，添加优化选项
        result = subprocess.run(['g++', 'code_craft.cpp', '-o', 'code_craft.exe', '-O2'], 
                              capture_output=True, text=True)
        if result.returncode != 0:
            print("编译错误输出:")
            print(result.stderr)
        return result.returncode == 0
    except Exception as e:
        print(f"编译错误: {e}")
        return False

def extract_score_from_result_file():
    """从result.txt文件中提取得分"""
    try:
        if os.path.exists('result.txt'):
            with open('result.txt', 'r') as f:
                content = f.read().strip()
                # print(f"读取到的result.txt内容: {content}")
                result = json.loads(content)
                # print(f"解析的JSON: {result}")
                if result["error_code"] == "interaction_successful":
                    # 将字符串格式的分数转换为浮点数
                    score = float(result["score"])
                    # print(f"成功提取分数: {score}")
                    return score
                else:
                    # print(f"错误代码不是successful: {result['error_code']}")
                    pass
        else:
            # print("result.txt 文件不存在")
            pass
        return 0
    except Exception as e:
        print(f"读取分数文件错误: {e}")
        return 0

def run_and_get_score(input_file):
    """运行程序并获取分数"""
    try:
        # 检查文件是否存在
        if not os.path.exists(input_file):
            print(f"错误：输入文件 {input_file} 不存在")
            return 0
            
        if not os.path.exists('code_craft.exe'):
            print("错误：code_craft.exe 不存在，请先编译代码")
            return 0
            
        if not os.path.exists('interactor/windows/interactor.exe'):
            print("错误：interactor.exe 不存在")
            return 0
            
        # 运行程序
        cmd = f'python run.py interactor/windows/interactor.exe {input_file} code_craft.exe'
        # print(f"执行命令: {cmd}")
        result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
        
        # 打印输出
        # print("程序输出:")
        # print(result.stdout)
        if result.stderr:
            print("错误输出:")
            print(result.stderr)
            
        # 检查result.txt是否存在
        if not os.path.exists('result.txt'):
            print("错误：result.txt 文件不存在")
            return 0
            
        # 读取结果文件
        with open('result.txt', 'r') as f:
            content = f.read().strip()
            # print(f"result.txt 内容: {content}")
            
        # 提取分数
        score = extract_score_from_result_file()
        # print(f"提取的分数: {score}")
        return score
        
    except Exception as e:
        print(f"运行程序时发生错误: {e}")
        return 0

def evaluate_params(params, runs=3, data_files=None):
    """评估一组参数的性能"""
    if data_files is None:
        data_files = ['sample.in']
    
    # 修改参数并编译代码
    modify_parameters(params)
    compile_success = False
    max_compile_retries = 3
    
    for attempt in range(max_compile_retries):
        if compile_code():
            compile_success = True
            break
        print(f"编译失败，尝试重新编译 ({attempt + 1}/{max_compile_retries})")
        time.sleep(2)  # 等待2秒后重试
    
    if not compile_success:
        print("编译失败，跳过评估")
        return 0
    
    all_scores = []
    for data_file in data_files:
        file_scores = []
        for _ in range(runs):
            score = run_and_get_score(data_file)
            if score > 0:
                file_scores.append(score)
                print(f"数据集 {data_file} 得分: {score}")
        
        # 计算该数据集的平均分数
        if file_scores:
            avg_score = np.mean(file_scores)
            all_scores.append(avg_score)
            print(f"数据集 {data_file} 平均得分: {avg_score}")
    
    # 返回所有数据集的平均分数
    final_score = np.mean(all_scores) if all_scores else 0
    print(f"总平均分数: {final_score}")
    return final_score

def evaluate_single_params(params, data_files):
    """评估单个参数组合的性能"""
    return evaluate_params(params, data_files=data_files)

def random_search(iterations=30, data_files=None):
    """随机搜索优化"""
    if data_files is None:
        data_files = ['sample.in']
    
    best_score = float('-inf')
    best_params = None
    candidates = []
    
    # 生成随机候选参数
    for _ in range(iterations):
        params = {}
        for param, (min_val, max_val) in PARAMS.items():
            if isinstance(min_val, int):
                params[param] = random.randint(min_val, max_val)
            else:
                params[param] = random.uniform(min_val, max_val)
        candidates.append(params)
    
    # 并行评估所有候选参数
    with concurrent.futures.ProcessPoolExecutor() as executor:
        futures = [executor.submit(evaluate_single_params, params, data_files) 
                  for params in candidates]
        scores = [future.result() for future in concurrent.futures.as_completed(futures)]
    
    # 找出最佳参数
    for params, score in zip(candidates, scores):
        if score > best_score:
            best_score = score
            best_params = params
            print(f"新的最佳分数: {best_score}")
            print(f"参数: {best_params}")
    
    return best_params, best_score

def bayesian_optimization(iterations=30, data_files=None):
    """贝叶斯优化"""
    if data_files is None:
        data_files = ['sample.in']
    
    try:
        from skopt import gp_minimize
        from skopt.space import Real, Integer
    except ImportError:
        print("请先安装scikit-optimize: pip install scikit-optimize")
        print("使用命令: pip install scikit-optimize")
        print("改用随机搜索方法...")
        return random_search(iterations=iterations, data_files=data_files)
    
    # 定义参数空间
    space = []
    param_names = []
    for param, (min_val, max_val) in PARAMS.items():
        param_names.append(param)
        if isinstance(min_val, int):
            space.append(Integer(min_val, max_val))
        else:
            space.append(Real(min_val, max_val))
    
    def objective(params):
        param_dict = dict(zip(param_names, params))
        return -evaluate_params(param_dict, data_files=data_files)  # 负号因为gp_minimize是最小化
    
    # 运行贝叶斯优化
    result = gp_minimize(
        objective,
        space,
        n_calls=iterations,
        n_random_starts=10,
        noise=0.1
    )
    
    # 转换结果为参数字典
    best_params = dict(zip(param_names, result.x))
    best_score = -result.fun  # 转换回正分数
    
    print(f"最佳分数: {best_score}")
    print(f"最佳参数: {best_params}")
    
    return best_params, best_score

if __name__ == "__main__":
    print("自动参数调优开始")
    print("可以在多个数据集上测试性能")
    
    # 询问是否在practice数据集上测试
    use_practice = input("是否在practice数据集上测试? (y/n): ").lower() == 'y'
    if use_practice:
        data_files = ['data/sample_practice.in']
    else:
        data_files = ['data/sample.in']
    
    print("\n1. 随机搜索")
    print("2. 贝叶斯优化 (需要安装scikit-optimize)")
    choice = input("请选择优化方法 (1/2): ")
    
    if choice == "1":
        iterations = int(input("请输入迭代次数 (默认30): ") or "30")
        best_params, best_score = random_search(iterations, data_files)
    elif choice == "2":
        iterations = int(input("请输入贝叶斯优化迭代次数 (默认30): ") or "30")
        best_params, best_score = bayesian_optimization(iterations, data_files)
    else:
        print("无效的选择，使用随机搜索方法")
        iterations = int(input("请输入迭代次数 (默认30): ") or "30")
        best_params, best_score = random_search(iterations, data_files)
    
    # 保存最佳参数
    with open('best_params.txt', 'w') as f:
        f.write(f"最佳分数: {best_score}\n")
        for param, value in best_params.items():
            f.write(f"{param}: {value}\n")
    
    print("\n调优完成！最佳参数已保存到 best_params.txt")
    print(f"最佳分数: {best_score}")
    print("最佳参数:")
    for param, value in best_params.items():
        print(f"{param}: {value}")
    
    # 应用最佳参数到code_craft.cpp
    modify_parameters(best_params)
    print("\n已更新code_craft.cpp中的参数")
    
    # 重新编译代码
    if compile_code():
        print("代码编译成功！")
    else:
        print("代码编译失败，请检查错误信息。") 
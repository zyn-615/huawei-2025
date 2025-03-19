from auto_tune import modify_parameters, compile_code, run_and_get_score
import subprocess
import os

def test_compilation():
    print("开始编译测试...")
    success = compile_code()
    if success:
        print("编译成功！")
        return True
    else:
        print("编译失败！")
        return False

def test_running():
    print("开始运行测试...")
    score = run_and_get_score('data/sample.in')
    print(f"运行得分: {score}")
    return score > 0

# 测试参数
test_params = {
    'JUMP_VISCOSITY': 1.0,
    'READ_ROUND_TIME': 10,
    'PRE_DISTRIBUTION_TIME': 5,
    'TEST_DENSITY_LEN': 5,
    'DISK_MIN_PASS': 5,
    'NUM_PIECE_QUEUE': 5
}

# 修改参数
print("修改参数...")
success = modify_parameters(test_params)

if success:
    print("参数修改成功！")
    
    # 测试编译
    if test_compilation():
        # 测试运行
        test_running()
else:
    print("参数修改失败！") 
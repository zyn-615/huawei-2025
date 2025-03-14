import matplotlib.pyplot as plt
import numpy as np

def read_input_file(filename):
    with open(filename, 'r') as f:
        # 读取基本参数 T, M, N, V, G
        T, M, N, V, G = map(int, f.readline().split())
        
        # 跳过3*M行频率数据
        for _ in range(3 * M):
            f.readline()
        
        # 初始化数据结构来存储每个时间片每个标签的读取次数
        tag_reads = {i: [0] * (T + 105) for i in range(1, M + 1)}
        
        # 读取每个时间片的操作
        current_time = 1
        write_tags = {}
        while current_time <= T + 105:
            # 时间戳
            timestamp = int(f.readline().split()[1])
            
            # 删除操作
            n_delete = int(f.readline())
            for _ in range(n_delete):
                f.readline()  # 跳过删除的对象ID
                
            # 写入操作
            n_write = int(f.readline())
            for _ in range(n_write):
                obj_id, _, obj_tag = map(int, f.readline().split())
                write_tags[obj_id] = obj_tag
                
            # 读取操作
            n_read = int(f.readline())
            for _ in range(n_read):
                _, obj_id = map(int, f.readline().split())
                # 只统计当前时间片写入的对象的读取
                if obj_id in write_tags:  # 保持这个检查
                    tag = write_tags[obj_id]
                    tag_reads[tag][current_time] += 1
            
            current_time += 1
            
    return T, M, tag_reads

def plot_tag_reads(T, M, tag_reads):
    plt.figure(figsize=(15, 6))  # 调整图形大小
    
    # 创建时间轴
    time_axis = np.arange(0, T+105, 1)
    
    # 为每个标签绘制折线
    for tag in range(1, M+1):
        reads = tag_reads[tag]
        # 使用更小的窗口来平滑曲线
        window_size = 50
        smoothed_reads = -np.convolve(reads, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(time_axis[window_size-1:], smoothed_reads, 
                color='blue',          # 蓝色线条
                linewidth=1.5,         # 线条粗细，调整为1.5
                linestyle='-',         # 实线
                marker='.',            # 小圆点作为标记
                markersize=3,          # 标记大小设为3
                markevery=200,         # 每200个点标记一次
                markerfacecolor='red', # 标记点为红色
                markeredgecolor='red', # 标记边缘为红色
                alpha=0.8              # 轻微透明度
        )
    
    # 设置坐标轴和网格
    plt.grid(True, linestyle='-', alpha=0.2)  # 浅色网格
    plt.xlim(0, T)                            # X轴范围
    
    # 设置刻度
    x_ticks = np.arange(0, T+1, 21600)        # 每21600一个刻度
    plt.xticks(x_ticks, x_ticks)
    
    # 移除上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 保存图片
    plt.savefig('read_patterns.png', 
                dpi=300,              # 高分辨率
                bbox_inches='tight',   # 自动调整边距
                pad_inches=0.1        # 边距大小
    )
    plt.close()

def main():
    filename = 'data/sample_practice.in'
    T, M, tag_reads = read_input_file(filename)
    plot_tag_reads(T, M, tag_reads)
    print("可视化结果已保存为 'read_patterns.png'")

if __name__ == "__main__":
    main()

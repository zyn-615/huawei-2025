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
        tag_size = {i: [0] * (T + 105) for i in range(1, M + 1)}
        
        # 读取每个时间片的操作
        current_time = 1
        write_tags = {}
        size_tags = {}
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
                obj_id, obj_size, obj_tag = map(int, f.readline().split())
                write_tags[obj_id] = obj_tag
                size_tags[obj_id] = obj_size  # 修复：存储对象大小
                
            # 读取操作
            n_read = int(f.readline())
            for _ in range(n_read):
                _, obj_id = map(int, f.readline().split())
                if obj_id in write_tags:
                    tag = write_tags[obj_id]
                    tag_reads[tag][current_time] += 1
                    tag_size[tag][current_time] += size_tags[obj_id]  # 修复：使用对象大小
            
            current_time += 1
            
    return T, M, tag_reads, tag_size

def plot_tag_reads_request_count(T, M, tag_reads):
    plt.figure(figsize=(15, 6))
    
    # 定义不同的颜色、线型和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 不同的颜色
    line_styles = ['-', '--', '-.', ':']  # 不同的线型
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']  # 不同的标记

    # 创建时间轴
    time_axis = np.arange(0, T+105, 1)
    
    # 为每个标签绘制折线
    for tag in range(1, M + 1):
        reads = tag_reads[tag]
        window_size = 50
        smoothed_reads = np.convolve(reads, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(time_axis[window_size-1:], smoothed_reads, 
                color=colors[(tag-1) % len(colors)],        # 循环使用颜色
                linewidth=1.5,
                linestyle=line_styles[(tag-1) % len(line_styles)],  # 循环使用线型
                marker=markers[(tag-1) % len(markers)],     # 循环使用标记
                markersize=4,
                markevery=500,                             # 减少标记密度
                label=f'标签 {tag}',                        # 添加图例标签
                alpha=0.8
        )
    
    # 设置坐标轴和网格
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.xlim(0, T)
    
    # 设置刻度
    x_ticks = np.arange(0, T+1, 21600)
    plt.xticks(x_ticks, x_ticks)
    
    # 移除上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1),  # 将图例放在图形右侧
              loc='upper left',
              borderaxespad=0.,
              frameon=False)              # 不显示图例边框
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('read_request_count_patterns.png', 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
    )
    plt.clf()
    plt.close()

def plot_tag_reads_request_size(T, M, tag_size):
    plt.figure(figsize=(15, 6))
    
    # 定义不同的颜色、线型和标记
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', 
              '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']  # 不同的颜色
    line_styles = ['-', '--', '-.', ':']  # 不同的线型
    markers = ['o', 's', '^', 'v', 'D', 'p', '*', 'h']  # 不同的标记

    # 创建时间轴
    time_axis = np.arange(0, T+105, 1)
    
    # 为每个标签绘制折线
    for tag in range(1, M + 1):
        reads = tag_size[tag]
        window_size = 50
        smoothed_reads = np.convolve(reads, np.ones(window_size)/window_size, mode='valid')
        
        plt.plot(time_axis[window_size-1:], smoothed_reads, 
                color=colors[(tag-1) % len(colors)],        # 循环使用颜色
                linewidth=1.5,
                linestyle=line_styles[(tag-1) % len(line_styles)],  # 循环使用线型
                marker=markers[(tag-1) % len(markers)],     # 循环使用标记
                markersize=4,
                markevery=500,                             # 减少标记密度
                label=f'标签 {tag}',                        # 添加图例标签
                alpha=0.8
        )
    
    # 设置坐标轴和网格
    plt.grid(True, linestyle='-', alpha=0.2)
    plt.xlim(0, T)
    
    # 设置刻度
    x_ticks = np.arange(0, T+1, 21600)
    plt.xticks(x_ticks, x_ticks)
    
    # 移除上边框和右边框
    plt.gca().spines['top'].set_visible(False)
    plt.gca().spines['right'].set_visible(False)
    
    # 添加图例
    plt.legend(bbox_to_anchor=(1.05, 1),  # 将图例放在图形右侧
              loc='upper left',
              borderaxespad=0.,
              frameon=False)              # 不显示图例边框
    
    # 调整布局以适应图例
    plt.tight_layout()
    
    # 保存图片
    plt.savefig('read_request_size_patterns.png', 
                dpi=300,
                bbox_inches='tight',
                pad_inches=0.1
    )
    plt.clf()
    plt.close()

def main():
    filename = 'data/sample_practice.in'
    T, M, tag_reads, tag_size = read_input_file(filename)
    plot_tag_reads_request_count(T, M, tag_reads)
    plot_tag_reads_request_size(T, M, tag_size)
    print("可视化结果已保存为 'read_request_count_patterns.png' 和 'read_request_size_patterns.png'")

if __name__ == "__main__":
    main()

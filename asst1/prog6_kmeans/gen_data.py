import numpy as np
import struct

def generate_mock_data(filename="./data/data.dat", M=1000000, N=100, K=3, epsilon=0.1):
    print(f"Generating {filename} with M={M}, N={N}, K={K}...")

    # 生成数据点
    centers = np.random.rand(K, N).astype(np.float64)
    true_labels = np.random.randint(0, K, size=M)   # 分配中心索引
    data = centers[true_labels] + np.random.normal(0, 0.5, size=(M, N))     # 叠加噪声
    data = data.astype(np.float64)

    # 初始质心 (稍微偏离数据点)
    cluster_centroids = (np.random.rand(K, N) * 0.1).astype(np.float64)
    
    # 随机分配
    cluster_assignments = np.random.randint(0, K, size=M).astype(np.int32)

    # 写入二进制文件
    with open(filename, "wb") as f:
        # 写入头信息: M, N, K (int32), epsilon (double64)
        f.write(struct.pack('iii', M, N, K))
        f.write(struct.pack('d', epsilon))
        
        # 写入数组
        f.write(data.tobytes())
        f.write(cluster_centroids.tobytes())
        f.write(cluster_assignments.tobytes())

    print(f"Successfully generated {filename}")

if __name__ == "__main__":
    generate_mock_data()
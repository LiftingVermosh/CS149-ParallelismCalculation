### SwiGLU
SwiGLU（"Swish门控线性单元"）是一种在现代大语言模型中使用的激活函数，作为早期门控线性单元的改进而引入。
SwiGLU的使用可见于包括许多其他近期Transformer架构在内的流行大语言模型。

SwiGLU(x, W, V, b, c, beta) = Swish(xW + b) ⊙ (xV + c)
其中 Swish(x) = x * sigmoid(beta * x)

输入
* `x`: 输入张量，形状为 [batch_size, seq_len, in_features]
* `W`: 权重矩阵，形状为 [in_features, hidden_size]
* `V`: 权重矩阵，形状为 [in_features, hidden_size]
* `b`: 偏置向量，形状为 [hidden_size,]
* `c`: 偏置向量，形状为 [hidden_size,]
* `beta`: Swish激活的标量值

输出
* 输出张量，形状为 [batch_size, seq_len, hidden_size]

参考文献：
* 原始SwiGLU论文见[此处](https://arxiv.org/pdf/1710.05941v1)。
# LoongFlow 配置与符号回归研究方案

> 项目目录：`/data/home/zyh/jaqmc`
>
> 目标：用 LoongFlow 调用大模型，寻找分数量子霍尔波函数的解析公式，并和现有 PySR 结果比较。

## 1. LoongFlow 放在哪里

建议把 LoongFlow 放在 `jaqmc` 旁边：

```text
/data/home/zyh/
├── jaqmc/                # 现有研究代码和数据
├── LoongFlow/            # LoongFlow 框架
└── loongflow-runs/       # LoongFlow 产生的日志和结果
```

不要把 LoongFlow 放进 `jaqmc/`，也不要让两个项目共用 `.venv`。

原因很简单：

- JaQMC 使用 JAX、PySR 和 Julia，环境比较复杂；
- LoongFlow 有自己的一套依赖；
- LoongFlow 会生成很多候选程序、日志和检查点；
- 分开放置后，升级或删除 LoongFlow 不会影响 JaQMC。

各类文件建议这样保存：

| 内容 | 保存位置 |
|---|---|
| LoongFlow 源码 | `/data/home/zyh/LoongFlow` |
| JaQMC 数据 | `/data/home/zyh/jaqmc/symbolic/datasets` |
| 符号回归任务配置 | `/data/home/zyh/jaqmc/symbolic/loongflow` |
| LoongFlow 运行结果 | `/data/home/zyh/loongflow-runs` |
| 大模型 API 密钥 | `~/.config/loongflow/llm.env` |

## 2. 安装 LoongFlow

### 第一步：克隆仓库

```bash
cd /data/home/zyh
git clone https://github.com/baidu-baige/LoongFlow.git
cd LoongFlow
```

### 第二步：创建独立环境

```bash
uv venv .venv --python 3.12
source .venv/bin/activate
```

### 第三步：安装

```bash
uv pip install -e .
```

注意最后有一个点 `.`，表示安装当前目录中的项目。

如果输入 `pip install -e .` 后出现 `pip: command not found`，不用安装传统 `pip`，继续使用：

```bash
uv pip install -e .
```

### 第四步：检查安装

```bash
python -c "import loongflow; print(loongflow.__file__)"
```

如果能打印出 LoongFlow 的文件路径，说明安装成功。

## 3. 准备大模型配置

LoongFlow 需要一个兼容 OpenAI 接口的大模型服务。可以使用在线 API，也可以使用自己部署的
vLLM 或 sglang 服务。

### 3.1 保存密钥

创建配置目录：

```bash
mkdir -p ~/.config/loongflow
chmod 700 ~/.config/loongflow
```

创建文件 `~/.config/loongflow/llm.env`，内容如下：

```bash
export LOONGFLOW_LLM_BASE_URL="https://你的服务地址/v1"
export LOONGFLOW_LLM_API_KEY="你的API密钥"
export LOONGFLOW_LLM_MODEL="openai/你的模型名称"
```

设置文件权限：

```bash
chmod 600 ~/.config/loongflow/llm.env
```

不要把 API 密钥直接提交到 Git。

## 4. 准备符号回归任务目录

在 JaQMC 中创建下面的目录：

```text
/data/home/zyh/jaqmc/symbolic/loongflow/
├── task_config.template.yaml   # 配置模板，不含真实密钥
├── task_config.yaml            # 实际配置，不提交到 Git
├── initial_program.py          # 大模型要改进的初始程序
└── eval_program.py             # 负责检查和评分
```

在 JaQMC 的 `.gitignore` 中加入：

```gitignore
/symbolic/loongflow/task_config.yaml
/symbolic/loongflow/*.rendered.yaml
```

## 5. LoongFlow 配置模板

把下面内容保存为：

```text
/data/home/zyh/jaqmc/symbolic/loongflow/task_config.template.yaml
```

```yaml
workspace_path: "/data/home/zyh/loongflow-runs/jaqmc-symbolic/default"

llm_config:
  url: "${LOONGFLOW_LLM_BASE_URL}"
  api_key: "${LOONGFLOW_LLM_API_KEY}"
  model: "${LOONGFLOW_LLM_MODEL}"
  temperature: 0.4
  context_length: 32768
  max_tokens: 6000
  top_p: 0.95
  timeout: 600

planners:
  evolve_planner:
    react_max_steps: 8

executors:
  evolve_executor_fuse:
    max_rounds: 3
    react_max_steps: 12
    score_threshold: 0.98

summarizers:
  evolve_summary:
    react_max_steps: 5

evolve:
  task: |
    Find a compact symbolic expression for a complex fractional-quantum-Hall
    wavefunction.

    Modify only the code between EVOLVE-BLOCK-START and EVOLVE-BLOCK-END.
    Keep the required function names and return values unchanged.

    Prefer short formulas that fit validation data well and satisfy fermionic
    exchange antisymmetry. Do not read the hidden test data.

  planner_name: "evolve_planner"
  executor_name: "evolve_executor_fuse"
  summary_name: "evolve_summary"
  max_iterations: 20
  target_score: 0.98
  concurrency: 1

  evaluator:
    timeout: 300

  database:
    storage_type: "in_memory"
    num_islands: 1
    population_size: 30
    checkpoint_interval: 1
    sampling_weight_power: 2
```

第一次测试不要直接运行几百次迭代。建议：

| 阶段 | 迭代次数 | 用途 |
|---|---:|---|
| 安装测试 | 2 | 检查模型和程序能否正常运行 |
| 小规模测试 | 20 | 调试任务和评分方法 |
| 正式实验 | 80–150 | 搜索最终公式 |

## 6. 生成实际配置

执行：

```bash
cd /data/home/zyh/jaqmc
source ~/.config/loongflow/llm.env

envsubst \
  '${LOONGFLOW_LLM_BASE_URL} ${LOONGFLOW_LLM_API_KEY} ${LOONGFLOW_LLM_MODEL}' \
  < symbolic/loongflow/task_config.template.yaml \
  > symbolic/loongflow/task_config.yaml

chmod 600 symbolic/loongflow/task_config.yaml
```

生成的 `task_config.yaml` 含有真实密钥，所以不能提交到 Git。

## 7. 大模型和评估程序分别做什么

整个过程可以简单理解为：

```text
大模型提出公式
      ↓
程序拟合公式中的常数
      ↓
在验证集上计算误差和物理约束
      ↓
LoongFlow 保存较好的公式
      ↓
大模型根据结果继续修改公式
```

大模型主要负责寻找公式结构，例如：

- 使用 `z1 - z2` 还是其他坐标组合；
- 使用几次幂；
- 是否包含共轭坐标；
- 是否需要径向修正项；
- 如何让公式满足电子交换反对称性。

固定的评估程序负责：

- 读取候选公式；
- 拒绝非法或过长的表达式；
- 在训练集上拟合数值系数；
- 在验证集上计算误差；
- 检查交换反对称性；
- 计算公式复杂度；
- 返回一个越大越好的总分。

大模型不应该直接读取测试集，也不应该修改 `eval_program.py`。

## 8. 建议的候选程序形式

`initial_program.py` 可以先使用非常简单的形式：

```python
import numpy as np


# EVOLVE-BLOCK-START
def wavefunction(z1, z2, c0_re=1.0, c0_im=0.0):
    return (c0_re + 1j * c0_im) * (z1 - z2)
# EVOLVE-BLOCK-END
```

这里：

- `z1` 和 `z2` 是复坐标，函数直接返回波函数值；
- 大模型修改 `return` 后的解析表达式，寻找更好的公式；
- 带数值默认值的附加参数由评估程序拟合；
- `c0_re` 和 `c0_im` 分别表示复系数的实部和虚部。

第一轮只允许这些运算：

```text
+  -  *  /  小整数次幂
```

先不要开放太多函数。只有简单多项式明显不够时，再加入 `exp` 等函数。

## 9. 数据应该怎样划分

必须把数据分成三部分：

| 数据 | 比例 | 用途 |
|---|---:|---|
| 训练集 | 64% | 拟合公式中的数值参数 |
| 验证集 | 16% | LoongFlow 比较候选公式 |
| 测试集 | 20% | 搜索结束后的最终检查 |

测试集不能参与大模型搜索。

当前 `symbolic/make_data.py` 已经保存了 `train_indices`、`test_indices`、
`source_indices` 和交换增强样本的来源关系。

但是当前 `symbolic/run_regression.py` 仍然会使用全部有效数据拟合。正式比较 PySR 和
LoongFlow 之前，需要先让 PySR 和 LoongFlow 使用完全相同的数据划分。

特别注意：一个原始样本和它交换电子后得到的增强样本，必须放在同一个数据分区中。否则
训练集和测试集会包含几乎相同的样本，测试误差会显得过分乐观。

## 10. 怎样评价一个公式

不要只看训练误差。至少记录下面几项：

| 指标 | 说明 |
|---|---|
| 验证集误差 | 公式对未参与参数拟合的数据是否准确 |
| 测试集误差 | 最终泛化能力 |
| Fidelity | 候选复波函数和目标波函数的重合程度 |
| 交换误差 | 交换两个同自旋电子后是否正确变号 |
| 公式复杂度 | 公式包含多少变量和运算 |
| 数值稳定性 | 是否产生 NaN、无穷大或异常大数值 |

LoongFlow 的总分可以使用下面的简单思想：

```text
误差越小       → 分数越高
物理约束越准确 → 分数越高
公式越短       → 分数越高
```

评估器必须同时保存各项原始指标，不能只保存最后的总分。

## 11. 如何运行

从 LoongFlow 目录启动，但任务文件仍保存在 JaQMC 中：

```bash
cd /data/home/zyh/LoongFlow
source .venv/bin/activate

export PYTHONPATH="/data/home/zyh/LoongFlow/src:/data/home/zyh/LoongFlow"

python agents/math_agent/math_evolve_agent.py \
  --config /data/home/zyh/jaqmc/symbolic/loongflow/task_config.yaml \
  --initial-file /data/home/zyh/jaqmc/symbolic/loongflow/initial_program.py \
  --eval-file /data/home/zyh/jaqmc/symbolic/loongflow/eval_program.py \
  --workspace-path /data/home/zyh/loongflow-runs/jaqmc-symbolic/n2_test \
  --max-iterations 2 \
  --log-level INFO
```

先运行 2 次迭代。确认以下内容正常后，再改为 20 次：

- 大模型接口连接成功；
- LoongFlow 能生成候选程序；
- 评估器能读取候选程序；
- 日志中出现有效分数；
- 检查点写入运行目录。

如果在集群计算节点运行，还需要确认计算节点可以访问大模型 API。如果计算节点没有外网，
需要使用集群内部可访问的 vLLM/sglang 服务。

## 12. 研究方案

### 阶段一：用已知公式做测试

先生成一个答案已知的简单数据集：

```text
F = C × (z1 - z2)³
```

目标不是发现新物理，而是检查整个系统是否可靠。

需要完成：

- [ ] 测试数据读取和划分；
- [ ] 测试复数系数拟合；
- [ ] 测试公式评分；
- [ ] 测试非法公式和超时处理；
- [ ] 使用至少 3 个随机种子；
- [ ] 检查是否都能找到与 `(z1-z2)³` 等价的公式。

如果这一阶段失败，不要急着增加迭代次数。应先检查评分函数、提示词和候选程序接口。

### 阶段二：研究现有 N=2 数据

使用：

```text
symbolic/datasets/disk_n2_Q6_kapa_1/dataset_50000.npz
```

比较三种方法：

1. 现有 PySR；
2. LoongFlow 不提示答案的搜索；
3. LoongFlow 加入物理约束的搜索。

建议分别做两种 LoongFlow 实验：

- **盲搜索**：只告诉模型坐标定义和物理要求，不告诉它答案是 `z12³`；
- **物理约束搜索**：先去掉已知高斯包络，再要求公式满足交换反对称性。

每种设置至少运行 3 个随机种子。只有多个种子都找到相近结构，结果才比较可信。

### 阶段三：研究更强相互作用

使用：

```text
symbolic/datasets/disk_n2_Q6_strength5/dataset_uniform_augmented.npz
symbolic/datasets/disk_n2_Q6_strength10/dataset_uniform_augmented.npz
```

不要直接搜索完整复杂波函数。建议按顺序进行：

1. 去掉已知高斯包络；
2. 提取最低阶反对称因子；
3. 只搜索剩下的修正项；
4. 如果简单多项式不够，再允许共轭坐标、径向变量和指数函数。

每个最终设置至少运行 5 个随机种子，并检查不同种子是否得到相同类型的修正项。

### 阶段四：扩展到 N=6

现有 PySR 结果还没有在 N=6 中恢复出清楚的多体结构。因此不要一开始就让大模型搜索完整
六电子波函数。

更合适的方法是：

- 先写入已知的交换反对称结构；
- 使用 Jastrow 因子、Slater 行列式或复合费米子基底；
- 让 LoongFlow 只搜索较简单的修正项；
- 在没有见过的电子构型和独立 checkpoint 上验证。

只有 N=2 的方法稳定后，才进入 N=6。

## 13. 最小实验表

| 数据 | 方法 | 建议随机种子数 |
|---|---|---:|
| 已知公式的合成数据 | LoongFlow 盲搜索、物理约束搜索 | 各 3 |
| N=2 基准数据 | PySR、LoongFlow 盲搜索、物理约束搜索 | 各 3 |
| 相互作用强度 5 | 表现最好的两种方法和 PySR | 各 5 |
| 相互作用强度 10 | 表现最好的两种方法和 PySR | 各 5 |
| N=6 | 先做小规模测试 | 各 2 |

## 14. 每次实验需要记录什么

每次运行至少记录：

- JaQMC 的 Git commit；
- LoongFlow 的 Git commit；
- 数据文件路径和哈希值；
- 数据划分随机种子；
- 搜索随机种子；
- 大模型名称；
- 完整任务配置，但要删除 API 密钥；
- 最佳候选公式；
- 验证集和测试集指标；
- 大模型调用次数和运行时间；
- 日志和 LoongFlow 检查点。

这样以后才能准确复现实验，也能判断改进来自大模型、数据划分还是评分方法。

## 15. 建议的实际执行顺序

按下面顺序执行最稳妥：

1. 安装 LoongFlow，确认 `import loongflow` 成功；
2. 配置大模型，运行官方示例 2 次迭代；
3. 创建 `symbolic/loongflow/` 下的三个任务文件；
4. 统一 PySR 和 LoongFlow 的数据划分；
5. 用已知的 `(z1-z2)³` 数据测试；
6. 对 N=2 基准数据运行 3 个随机种子；
7. 研究相互作用强度 5 和 10；
8. 最后再研究 N=6。

## 16. 参考资料

- [LoongFlow 仓库](https://github.com/baidu-baige/LoongFlow)
- [LoongFlow Math Agent 示例](https://github.com/baidu-baige/LoongFlow/tree/main/agents/math_agent/examples)
- [官方圆堆积配置示例](https://github.com/baidu-baige/LoongFlow/blob/main/agents/math_agent/examples/packing_circle_in_unit_square/task_config.yaml)
- [Math Agent 入口程序](https://github.com/baidu-baige/LoongFlow/blob/main/agents/math_agent/math_evolve_agent.py)

LoongFlow 目前没有现成的符号回归示例。这里使用的是它的通用方式：让大模型修改候选程序，
再由固定评估器检查结果和返回分数。

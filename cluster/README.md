# 人工双重认证与单卡短任务

在自己的终端完成 SSH 双重认证。仓库不保存账号、主机或凭据。以下命令在 Linux 登录节点执行；GPU 计算只经 sbatch 在计算节点执行。

## 只读检查

```bash
sinfo -o '%P %a %l %D %t %G'
squeue --me -o '%.18i %.14P %.22j %.10T %.10M %.10l %R'
module load Anaconda3/2025.06
module list 2>&1
command -v python
python --version
conda env list
id
for project_group in $(id -Gn); do
  for project_disk in /data1 /data2; do
    project_dir="$project_disk/$project_group/$(id -un)"
    if [ -d "$project_dir" ]; then
      ls -ld "$project_dir"
      if [ -w "$project_dir" ]; then
        printf 'Writable project storage: %s\n' "$project_dir"
      fi
    fi
  done
done
```

学院文档规定数据目录为 `/data2/用户组/用户名`，而非 `/data2/用户名`。上面逐一检查所属组，避免多个组导致路径拼接错误。若未列出数据目录，回传输出后继续核对，不在共享根目录自行创建目录。模块只改变当前登录 shell，不修改已有作业。

回传这些结果后选择真实分区、项目目录和环境。不要假设分区名是 4090/L40，不改动其他作业。[存储说明](https://saids.hpc.gleamoe.com/guideline/storage/)、[软件说明](https://saids.hpc.gleamoe.com/guideline/software/)。

## 环境

在已确认属于自己的数据目录中 clone 项目并进入仓库。准备独立的 CUDA Python 环境；本机 CPU 环境不能直接复制。依赖安装与模型下载先完成，避免耗尽 GPU 探测时限。

```bash
python -c 'import importlib.metadata as m; import torch; print(m.version("torch"), torch.version.cuda)'
export PYTHON_BIN="$(command -v python)"
```

这里只检查 PyTorch 构建；登录节点无 GPU 不代表计算节点 CUDA 不可用。没有 torch 时先确认模块与驱动兼容性，再选择安装版本。

## 预计开始时间与提交

从 sinfo 选择分区，设置 PARTITION（去掉分区名末尾默认标记 `*`）。运行：

```bash
bash cluster/check_queue.sh "$PARTITION"
```

`sbatch --test-only` 做提交检查和调度估计，不真实排队；时间可能缺失或变化。然后仅提交一个任务：

也可在准备代码之前比较已确认存在的 RTX4090 和 L40 分区，资源与默认探测一致：

```bash
sbatch --test-only --partition=RTX4090 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:10:00 --wrap='true'
sbatch --test-only --partition=L40 --nodes=1 --ntasks=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G --time=00:10:00 --wrap='true'
```

准备完成后才执行下方实际提交命令。示例中的 `<JOB_ID>` 不能原样粘贴，尖括号会被 Bash 当作重定向；下方自动保存真实作业号：

```bash
job_id=$(sbatch --parsable --partition="$PARTITION" cluster/probe_gpu.slurm)
job_id=${job_id%%;*}
printf 'Submitted job: %s\n' "$job_id"
squeue --start --jobs="$job_id"
```

默认 1 GPU、4 CPU、32GB、10 分钟，检查 CUDA 和小矩阵计算。完成后回传：

```bash
sacct -j "$job_id" --format=JobID,JobName,Partition,State,Submit,Start,End,Elapsed,Timelimit,ExitCode
cat ".tmp/cluster/probe-$job_id/probe.json"
cat "lgrag-probe-$job_id.err"
```

自定义 RUN_ROOT 时结果在该目录的 probe-JOB_ID 下。用 Slurm 最终状态和程序输出共同确认成功。

## 可选真实 BGE 模型探测

配置 HF_HOME 到自己的数据目录，提前下载：

```bash
hf download BAAI/bge-small-zh-v1.5
hf download BAAI/bge-reranker-base
```

项目依赖准备好后 `export PROBE_MODELS=1`，再提交同一脚本。只读缓存，编码两段文本、重排两个候选；缺模型或 CUDA 不可用即失败。普通硬件探测成功不代表模型路径成功。

正式评测见 [外部对比说明](../docs/external_comparison.md)。首轮单作业 30–60 分钟，先确认数据标注与 BGE/LlamaIndex 路径。Qwen3 与 LightRAG 尚待适配，不预提交长任务。

参考：[学院 Slurm](https://saids.hpc.gleamoe.com/guideline/slurm/)、[sbatch](https://slurm.schedmd.com/sbatch.html)、[squeue](https://slurm.schedmd.com/squeue.html)。

# 缓存完成后训练停顿：2026-09-21 排查记录

这次停顿发生在训练集缓存发布之后、验证集缓存的 DataLoader 子进程启动阶段；现有证据不支持直接认定为训练集 persistent worker 析构死锁。

## 观察到的证据

- 最后一条终端日志是训练集 246,913/246,913 帧缓存提取完成。
- 磁盘已存在完整 train 缓存及 manifest；另一个临时构建目录的数组尺寸对应验证集 28,359 帧。说明程序已从 train 缓存函数返回并进入 val 缓存。
- 主进程 166647 的等待位置是 `pipe_write`。
- 新子进程中两个处于 DataLoader 等待状态，另一个 PID 170076 为僵尸进程，其 `/proc/<pid>/stat` 最后字段 exit_code=139，对应 SIGSEGV。
- ptrace 权限不允许附加 strace，未取得崩溃线程的 Python/C 栈。因此不能进一步断言是 OpenBLAS、PyTorch、OpenCV 或硬件中的某一项导致段错误。

这些证据表明：至少一个验证集 worker 启动时异常退出，父进程同时停在管道写入，符合 spawn 启动/传递数据阶段卡住的现象。仅看到最后一条进度日志不足以确定程序仍停在 train 缓存。

## 本次修改

1. `prepare_features` 以配置副本强制 `persistent_workers=False`。缓存每个 split 只遍历一次，不需要保留 worker，且不改变其他训练 DataLoader 的设置。
2. 本机默认 `cache_num_workers=0`，缓存提取在主进程执行，绕过本次出错的 spawn 路径。正式缓存头训练原本也使用 `cached_num_workers=0`。
3. 增加开始提取、迭代结束、刷盘、校验、发布和载入完成日志，区分计算进度与整个缓存阶段完成。
4. 新增 `scripts/train.py --cache_num_workers` 覆盖项。以后排查清楚子进程崩溃后可显式恢复并行提取；不能把恢复并行视作已验证安全。
5. 用 SIGINT 停止了原阻塞训练并确认其子进程退出，保留已发布 train 缓存；重新训练复用其校验通过的特征，只重新提取 val。

## 使用

```bash
conda activate value_function
python scripts/train.py
# 或明确使用不启动缓存子进程的模式
python scripts/train.py --cache_num_workers 0
```

沿用当前工程命名，模型保存为 `checkpoints/optimized/best_model.pt`，不是 `best.pt`。诊断日志保存在忽略提交的 `artifacts/performance/cache-recovery-training.log`。

性能取舍：关闭缓存 worker 可能降低首次视频解码吞吐，但不会反复重建已有匹配缓存，也不影响缓存建好之后的回归头/分布头训练并行方式。本次只针对阻塞进行修复，未更改回报、数据划分、网络分布定义或 checkpoint 架构标记。

## 恢复验证结果

- 21 项单元/命令行测试通过。
- train 缓存直接复用，val 28,359 帧以约 216 帧/秒提取完成；缓存发布后正常进入训练。
- 按原训练预算运行，第 14 轮、3,388 次参数更新后触发早停，正常退出（退出码 0）。
- 已生成 `checkpoints/optimized/best_model.pt`、`config.json`、`metrics.json`，不是仅执行冒烟测试。

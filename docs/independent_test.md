# 当前 checkpoint 的独立测试

完整测试37条轨迹/34,759帧，模型测试MSE **0.02746**；训练集拟合的批次+帧序号对照为 **0.02250**。目前不能确认可直接用于RECAP优势标注。

[详细报告](../artifacts/evaluation/checkpoint-7516692c9def-test/report.md) · [交互浏览](../artifacts/evaluation/checkpoint-7516692c9def-test/index.html)

评估固定第16轮checkpoint，未更新模型或原始数据。17项测试复测通过；独立重放全测试预测逐元素一致。复现命令、置信区间、分组指标和运行时异常记录见详细报告。

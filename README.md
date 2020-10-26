# Material-Group 

数据集：已生成的模拟数据 material_group2.xlsx

和风险相关的24个指标

和重要性相关的9个指标

material_group_data_generate.ipynb为模拟数据生成器；
material_group_classification.ipynb为物资分类器

1）使用逻辑回归进行建模

四种分类：战略、瓶颈、一般、杠杆

2）超参数调优（贝叶斯优化）

针对逻辑回归模型中的C和max_iter进行超参数优化

3）封装一套自己的tools，用于类似任务，存在woe_tools文件夹下，工具包为woe_tools.py，可用test_demo.py对工具包的使用进行测试。

待更新：绘制BCG矩阵四象限，根据money_label和risk_label绘制Kraljic采购定位模型，将物资分类为四个类：战略物资（Strategic Items）[money_label=1, risk_label=1]、瓶颈物资（Bottleneck Items）[money_label=0, risk_label=1]、杠杆物资（Strategic Items）[money_label=1, risk_label=0]、一般物资（Non-Critical Items）[money_label=0, risk_label=0]



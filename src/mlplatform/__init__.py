"""mlplatform —— 项目隔离的多阶段计算机视觉训练平台。

四个阶段（每个阶段跑在 envs/ 下**独立**的 conda 环境里）：

    自动标注 / 数据生成  ->  label_system
    模型训练             ->  train_system
    模型评估 / 推理测试   ->  test_system
    平台 Web 服务         ->  main_system

项目、数据集、模型、训练记录、评估记录全部按项目隔离在 ``projects/<project>/`` 下，
迁移/备份一个项目只需要拷贝那一个目录。
"""

__version__ = "2.0.0"

# poisonswarm_defeat

攻击特征检测原型（defense prototype），配套研究项目 PoisonSwarm / FCGen（LLM 安全红队：众包式分布式有害信息生成）。

- semantic_parser.py —— 攻击特征的语义解析
- defense_controller.py —— 分档熔断 + 多模型共识的防御控制器
- thresholds.yaml —— 风险分档阈值配置
- test_attack_chains.py —— 攻击链测试用例
- bayesian_intent.py —— 会话级意图的贝叶斯建模与风险评估（含历史会话追踪）
- main_defender.py —— 入口

运行：

pip install -r requirements.txt
python main_defender.py

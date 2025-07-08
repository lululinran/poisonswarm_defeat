from pgmpy.models import BayesianNetwork
from pgmpy.factors.discrete import TabularCPD
from pgmpy.inference import VariableElimination

class BayesianIntentModel:
    def __init__(self):
        # 构建贝叶斯网络
        self.model = BayesianNetwork([
            ('GlobalIntent', 'RiskScore'),
            ('Counterfactual', 'RiskScore'),
            ('ComponentChain', 'RiskScore'),
            ('Progression', 'RiskScore'),
            ('CriticalUnits', 'RiskScore')
        ])
        
        # 定义条件概率分布
        self._define_cpds()
    
    def _define_cpds(self):
        # 全局意图先验概率 (良性/可疑/恶意)
        cpd_global = TabularCPD(
            'GlobalIntent', 3, 
            [[0.70], [0.25], [0.05]],  # 70%良性，25%可疑，5%恶意
            state_names={'GlobalIntent': ['benign', 'suspicious', 'malicious']}
        )
        
        # 反事实标记(CM)条件概率
        cpd_counterfact = TabularCPD(
            'Counterfactual', 2,
            [[0.95, 0.65, 0.20],  # 当全局意图为良性时，95%无CM
             [0.05, 0.35, 0.80]], # 当全局意图为恶意时，80%有CM
            evidence=['GlobalIntent'],
            evidence_card=[3],
            state_names={
                'Counterfactual': ['false', 'true'],
                'GlobalIntent': ['benign', 'suspicious', 'malicious']
            }
        )
        
        # PoisonSwarm论文中的风险权重
        cpd_risk = TabularCPD(
            'RiskScore', 3,  # 低/中/高风险
            [
                [0.95, 0.70, 0.35, 0.20, 0.10, 0.05, 0.01, 0.01],  # 低风险
                [0.04, 0.25, 0.40, 0.30, 0.30, 0.25, 0.15, 0.10],  # 中风险
                [0.01, 0.05, 0.25, 0.50, 0.60, 0.70, 0.84, 0.89]   # 高风险
            ],
            evidence=['GlobalIntent', 'Counterfactual', 'ComponentChain', 'Progression', 'CriticalUnits'],
            evidence_card=[3, 2, 2, 2, 2],
            state_names={'RiskScore': ['low', 'medium', 'high']}
        )
        
        self.model.add_cpds(cpd_global, cpd_counterfact, cpd_risk)
    
    def evaluate_risk(self, evidence: dict) -> dict:
        """基于证据评估风险"""
        infer = VariableElimination(self.model)
        
        # 转换证据格式
        pgmpy_evidence = {}
        for key, value in evidence.items():
            if key == 'GlobalIntent':
                pgmpy_evidence[key] = value
            else:
                pgmpy_evidence[key] = 'true' if value else 'false'
        
        # 计算风险分布
        risk_dist = infer.query(['RiskScore'], evidence=pgmpy_evidence)
        intent_dist = infer.query(['GlobalIntent'], evidence=pgmpy_evidence)
        
        return {
            "risk_level": risk_dist.values.argmax(),
            "risk_distribution": {
                "low": float(risk_dist.values[0]),
                "medium": float(risk_dist.values[1]),
                "high": float(risk_dist.values[2])
            },
            "intent_distribution": {
                "benign": float(intent_dist.values[0]),
                "suspicious": float(intent_dist.values[1]),
                "malicious": float(intent_dist.values[2])
            }
        }
    
    def update_belief(self, new_evidence: dict):
        """动态更新信念网络"""
        # 简单实现 - 实际应调整CPD参数
        pass

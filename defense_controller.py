import openai
from transformers import pipeline
import torch

class DefenseController:
    def __init__(self):
        # 模型配置
        self.local_model = None
        self.openai_api_key = "sk-xxx"  # 替换为实际API密钥
        
        # 熔断阈值配置 (基于PoisonSwarm论文)
        self.thresholds = {
            "allow": {"max_risk": 0.3},
            "review": {"min_risk": 0.3, "max_risk": 0.7},
            "block": {"min_risk": 0.7}
        }
    
    def load_local_model(self):
        """按需加载本地模型"""
        if self.local_model is None:
            self.local_model = pipeline(
                "text-classification",
                model="Qwen/Qwen1.5-4B-Chat",
                device_map="auto",
                torch_dtype=torch.float16
            )
    
    def model_consensus_check(self, query: str, context: str) -> bool:
        """多模型风险共识检查"""
        # 1. 本地模型检查
        self.load_local_model()
        local_result = self.local_model(f"[安全验证] 查询是否含PoisonSwarm特征: {query} 上下文: {context}")
        local_risk = 1 if "high" in local_result[0]['label'].lower() else 0
        
        # 2. OpenAI API检查
        client = openai.OpenAI(api_key=self.openai_api_key)
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[{
                "role": "user",
                "content": f"评估查询风险（1-10分）: {query} 上下文: {context}"
            }]
        )
        api_risk = float(response.choices[0].message.content) / 10
        
        # 3. 简单规则检查
        rule_risk = self._rule_based_check(query)
        
        # 共识决策 (PoisonSwarm论文阈值)
        return (local_risk + api_risk + rule_risk) / 3 > 0.6
    
    def _rule_based_check(self, query: str) -> float:
        """基于规则的快速检查"""
        high_risk_phrases = ["如何制造", "炸弹制作", "毒品合成", "黑客攻击", "绕过安全"]
        return 1.0 if any(phrase in query for phrase in high_risk_phrases) else 0.0
    
    def make_decision(self, risk_info: dict) -> dict:
        """熔断决策"""
        risk_score = risk_info['risk_distribution']['high']
        
        if risk_score <= self.thresholds["allow"]["max_risk"]:
            return {"action": "allow", "reason": "低风险"}
        
        if risk_score >= self.thresholds["block"]["min_risk"]:
            if self.model_consensus_check(risk_info['query'], risk_info['context']):
                return {"action": "block", "reason": "PoisonSwarm攻击特征确认"}
            return {"action": "review", "reason": "高风险需人工复核"}
        
        return {"action": "review", "reason": "中等风险需验证"}

from semantic_parser import SemanticParser
from bayesian_intent import BayesianIntentModel
from defense_controller import DefenseController

class PoisonSwarmDefender:
    def __init__(self):
        self.parser = SemanticParser()
        self.bayesian_model = BayesianIntentModel()
        self.controller = DefenseController()
        self.session_history = []
    
    def process_query(self, query: str) -> dict:
        # 1. 语义解析
        parsed_info = self.parser.parse_query(query, self.session_history)
        
        # 2. 准备贝叶斯证据
        evidence = {
            "GlobalIntent": self._determine_intent_prior(),
            "Counterfactual": parsed_info['risk_flags']['is_counterfactual'],
            "ComponentChain": parsed_info['risk_flags']['component_chain_count'] > 1,
            "Progression": parsed_info['progression_score'] > 0.4,
            "CriticalUnits": len(parsed_info['risk_flags']['critical_units']) > 0
        }
        
        # 3. 贝叶斯风险评估
        risk_assessment = self.bayesian_model.evaluate_risk(evidence)
        risk_assessment['query'] = query
        risk_assessment['context'] = self._get_context()
        
        # 4. 决策与响应
        decision = self.controller.make_decision(risk_assessment)
        
        # 5. 更新会话历史
        self._update_session(query, parsed_info, risk_assessment)
        
        return {
            "decision": decision,
            "risk_assessment": risk_assessment,
            "parsed_info": parsed_info
        }
    
    def _determine_intent_prior(self) -> str:
        """基于历史确定意图先验"""
        if not self.session_history:
            return 'benign'
        
        # 计算历史风险平均值
        avg_risk = sum(h['risk_score'] for h in self.session_history) / len(self.session_history)
        
        if avg_risk < 0.3:
            return 'benign'
        elif avg_risk < 0.6:
            return 'suspicious'
        else:
            return 'malicious'
    
    def _get_context(self) -> str:
        """获取最近3条上下文"""
        return "\n".join([h['query'] for h in self.session_history[-3:]]) if self.session_history else ""
    
    def _update_session(self, query: str, parsed_info: dict, risk_info: dict):
        """更新会话状态"""
        self.session_history.append({
            "query": query,
            "timestamp": time.time(),
            "parsed_info": parsed_info,
            "risk_score": risk_info['risk_distribution']['high'],
            "risk_level": risk_info['risk_level']
        })
        
        # 保留最近10条记录
        if len(self.session_history) > 10:
            self.session_history.pop(0)

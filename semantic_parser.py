import re
import json

class SemanticParser:
    def __init__(self):
        # PoisonSwarm特征库
        self.cm_triggers = ["假设", "如果", "虚构", "小说", "剧本", "研究", "学术"]
        self.act_indicators = ["步骤", "方法", "如何制作", "怎样合成", "过程", "配方"]
        self.high_risk_units = ["爆炸物", "毒品", "武器", "黑客", "恶意软件", "硝酸甘油", "剧毒"]
    
    def parse_query(self, query: str, history: list) -> dict:
        """深度解析查询语义结构"""
        # 检测反事实标记(CM)
        is_counterfactual = any(trigger in query for trigger in self.cm_triggers)
        
        # 检测组件链式请求(ACT)
        component_chain_count = sum(1 for indicator in self.act_indicators if indicator in query)
        
        # 提取实体
        entities = {
            "action": self._extract_actions(query),
            "objects": self._extract_objects(query),
            "goal": self._extract_goal(query)
        }
        
        # 高风险语义单元
        critical_units = [unit for unit in self.high_risk_units if unit in query]
        
        return {
            "entities": entities,
            "risk_flags": {
                "is_counterfactual": is_counterfactual,
                "component_chain_count": component_chain_count,
                "critical_units": critical_units
            },
            "progression_score": self._calc_progression(query, history)
        }
    
    def _extract_actions(self, query: str) -> list:
        """提取动作动词"""
        actions = ["制造", "合成", "制作", "获取", "购买", "使用", "攻击", "破坏"]
        return [word for word in actions if word in query]
    
    def _extract_objects(self, query: str) -> list:
        """提取对象名词"""
        # 简化的实体识别，实际应使用NER模型
        pattern = r'(\w+的)?(性质|成分|制作|合成|获取|购买)'
        matches = re.findall(pattern, query)
        return list(set([match[0].strip("的") for match in matches if match[0]]))
    
    def _extract_goal(self, query: str) -> str:
        """识别最终应用场景"""
        goal_keywords = ["用于", "用来", "应用", "目的是", "目标是"]
        for keyword in goal_keywords:
            if keyword in query:
                return query.split(keyword)[-1].strip()
        return ""
    
    def _calc_progression(self, query: str, history: list) -> float:
        """计算目标硬化程度（0-1）"""
        if not history:
            return 0.0
        
        # 计算当前查询与前3次查询的风险差异
        prev_risk_score = sum(h.get("progression_score", 0) for h in history[-3:])
        current_risk = len(self._extract_objects(query)) * 0.2
        
        progression = max(0, min(1, current_risk - prev_risk_score))
        return round(progression, 2)

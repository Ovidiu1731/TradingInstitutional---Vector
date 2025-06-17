import json
import os
from collections import defaultdict, Counter
from datetime import datetime, timedelta
import re
from typing import Dict, List, Any
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeedbackAnalyzer:
    def __init__(self, feedback_log_path: str = "feedback_log.jsonl"):
        self.feedback_log_path = feedback_log_path
        
    def load_feedback_data(self) -> List[Dict]:
        """Load all feedback data from the log file."""
        feedback_data = []
        if not os.path.exists(self.feedback_log_path):
            logger.warning(f"Feedback log file {self.feedback_log_path} not found")
            return feedback_data
            
        try:
            with open(self.feedback_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    line = line.strip()
                    if line:
                        feedback_data.append(json.loads(line))
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
            
        return feedback_data
    
    def analyze_feedback_patterns(self) -> Dict[str, Any]:
        """Analyze feedback patterns to identify improvement areas."""
        feedback_data = self.load_feedback_data()
        
        if not feedback_data:
            return {"error": "No feedback data available"}
        
        analysis = {
            "total_feedback": len(feedback_data),
            "feedback_distribution": Counter(),
            "problem_questions": [],
            "successful_questions": [],
            "common_issues": defaultdict(list),
            "query_type_performance": defaultdict(lambda: {"positive": 0, "negative": 0, "neutral": 0}),
            "recent_trends": self._analyze_recent_trends(feedback_data),
            "improvement_suggestions": []
        }
        
        # Analyze each feedback entry
        for entry in feedback_data:
            feedback_type = entry.get("feedback", "unknown")
            question = entry.get("question", "")
            answer = entry.get("answer", "")
            query_type = entry.get("query_type", "unknown")
            
            # Count feedback types
            analysis["feedback_distribution"][feedback_type] += 1
            
            # Track query type performance
            analysis["query_type_performance"][query_type][feedback_type] += 1
            
            # Identify problematic vs successful questions
            if feedback_type == "negative":
                analysis["problem_questions"].append({
                    "question": question,
                    "answer": answer[:200] + "...",
                    "timestamp": entry.get("timestamp", "")
                })
                
                # Identify common issues in negative feedback
                self._categorize_issues(question, answer, analysis["common_issues"])
                
            elif feedback_type == "positive":
                analysis["successful_questions"].append({
                    "question": question,
                    "answer": answer[:200] + "...",
                    "timestamp": entry.get("timestamp", "")
                })
        
        # Generate improvement suggestions
        analysis["improvement_suggestions"] = self._generate_suggestions(analysis)
        
        return analysis
    
    def _analyze_recent_trends(self, feedback_data: List[Dict]) -> Dict[str, Any]:
        """Analyze trends in recent feedback (last 7 days)."""
        recent_cutoff = datetime.now() - timedelta(days=7)
        recent_feedback = []
        
        for entry in feedback_data:
            try:
                timestamp = datetime.fromisoformat(entry.get("timestamp", ""))
                if timestamp >= recent_cutoff:
                    recent_feedback.append(entry)
            except ValueError:
                continue
        
        if not recent_feedback:
            return {"message": "No recent feedback available"}
        
        recent_distribution = Counter(entry.get("feedback", "unknown") for entry in recent_feedback)
        satisfaction_rate = (recent_distribution["positive"] / len(recent_feedback)) * 100
        
        return {
            "total_recent": len(recent_feedback),
            "distribution": dict(recent_distribution),
            "satisfaction_rate": round(satisfaction_rate, 2)
        }
    
    def _categorize_issues(self, question: str, answer: str, issues: Dict):
        """Categorize common issues from negative feedback."""
        question_lower = question.lower()
        answer_lower = answer.lower()
        
        # Define issue patterns
        issue_patterns = {
            "incomplete_answer": [
                "nu am gasit", "nu am putut", "nu stiu", "informații insuficiente"
            ],
            "liquidity_issues": [
                "lichiditate", "liq", "hod", "lod"
            ],
            "setup_confusion": [
                "setup", "og", "tg", "tcg", "fvg"
            ],
            "mss_problems": [
                "mss", "market structure", "structura"
            ],
            "robotic_language": [
                "este important sa", "este necesar sa", "este esential sa"
            ],
            "missing_context": [
                "nu am suficiente", "context insuficient", "mai multe detalii"
            ]
        }
        
        for issue_type, patterns in issue_patterns.items():
            if any(pattern in answer_lower for pattern in patterns):
                issues[issue_type].append({
                    "question": question,
                    "answer_snippet": answer[:100] + "..."
                })
    
    def _generate_suggestions(self, analysis: Dict) -> List[str]:
        """Generate actionable improvement suggestions based on analysis."""
        suggestions = []
        
        # Suggestion based on feedback distribution
        negative_rate = analysis["feedback_distribution"]["negative"] / analysis["total_feedback"]
        if negative_rate > 0.3:
            suggestions.append("High negative feedback rate (>30%) - review system prompts and retrieval quality")
        
        # Suggestions based on common issues
        common_issues = analysis["common_issues"]
        
        if "incomplete_answer" in common_issues and len(common_issues["incomplete_answer"]) > 2:
            suggestions.append("Frequently incomplete answers - consider lowering similarity thresholds or improving embeddings")
        
        if "liquidity_issues" in common_issues:
            suggestions.append("Liquidity-related questions causing issues - review liquidity content in vector database")
        
        if "robotic_language" in common_issues:
            suggestions.append("Robotic language detected - update system prompts for more natural responses")
        
        if "setup_confusion" in common_issues:
            suggestions.append("Setup-related confusion - consider adding more specific examples in system prompt")
        
        # Query type specific suggestions
        for query_type, performance in analysis["query_type_performance"].items():
            total = sum(performance.values())
            if total > 5:  # Only analyze types with sufficient data
                negative_rate = performance["negative"] / total
                if negative_rate > 0.4:
                    suggestions.append(f"Query type '{query_type}' has high negative feedback - review handling logic")
        
        return suggestions
    
    def generate_improvement_report(self) -> str:
        """Generate a comprehensive improvement report."""
        analysis = self.analyze_feedback_patterns()
        
        if "error" in analysis:
            return "No feedback data available for analysis."
        
        report = f"""
# AI Model Feedback Analysis Report
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

## Summary
- Total Feedback Entries: {analysis['total_feedback']}
- Positive: {analysis['feedback_distribution']['positive']} ({analysis['feedback_distribution']['positive']/analysis['total_feedback']*100:.1f}%)
- Negative: {analysis['feedback_distribution']['negative']} ({analysis['feedback_distribution']['negative']/analysis['total_feedback']*100:.1f}%)
- Neutral: {analysis['feedback_distribution']['neutral']} ({analysis['feedback_distribution']['neutral']/analysis['total_feedback']*100:.1f}%)

## Recent Trends (Last 7 Days)
{analysis['recent_trends']}

## Common Issues Identified
"""
        
        for issue_type, cases in analysis["common_issues"].items():
            if cases:
                report += f"### {issue_type.replace('_', ' ').title()}\n"
                report += f"- Occurrences: {len(cases)}\n"
                for case in cases[:3]:  # Show top 3 examples
                    report += f"  - Q: {case['question'][:50]}...\n"
                report += "\n"
        
        report += "## Improvement Suggestions\n"
        for suggestion in analysis["improvement_suggestions"]:
            report += f"- {suggestion}\n"
        
        return report

if __name__ == "__main__":
    analyzer = FeedbackAnalyzer()
    report = analyzer.generate_improvement_report()
    print(report) 
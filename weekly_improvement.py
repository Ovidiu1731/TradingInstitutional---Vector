#!/usr/bin/env python3
"""
Weekly AI Improvement Script
Automatically analyzes feedback and suggests system improvements
"""

import os
import sys
import json
import smtplib
from email.mime.text import MimeText
from email.mime.multipart import MimeMultipart
from datetime import datetime, timedelta
import logging

# Add current directory to path to import local modules
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feedback_analyzer import FeedbackAnalyzer
from prompt_optimizer import PromptOptimizer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('weekly_improvement.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class WeeklyImprovementEngine:
    def __init__(self):
        self.analyzer = FeedbackAnalyzer()
        self.optimizer = PromptOptimizer()
        
    def run_weekly_analysis(self):
        """Run the complete weekly improvement analysis"""
        logger.info("Starting weekly AI improvement analysis...")
        
        try:
            # 1. Analyze feedback patterns
            analysis = self.analyzer.analyze_feedback_patterns()
            
            if "error" in analysis:
                logger.warning("No feedback data available for analysis")
                return {"status": "no_data", "message": "No feedback data available"}
            
            # 2. Generate improvement report
            report = self.analyzer.generate_improvement_report()
            
            # 3. Generate optimized prompts
            optimized_prompt = self.optimizer.generate_optimized_prompt()
            improvements_summary = self.optimizer.get_prompt_improvements_summary()
            
            # 4. Save artifacts
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # Save improvement report
            report_filename = f"improvement_report_{timestamp}.md"
            with open(report_filename, 'w', encoding='utf-8') as f:
                f.write(report)
            
            # Save optimized prompt
            prompt_filename = f"optimized_prompt_{timestamp}.txt"
            with open(prompt_filename, 'w', encoding='utf-8') as f:
                f.write(optimized_prompt)
            
            # Save summary JSON
            summary_filename = f"improvement_summary_{timestamp}.json"
            summary_data = {
                "analysis": analysis,
                "improvements_summary": improvements_summary,
                "generated_at": datetime.now().isoformat(),
                "files_generated": {
                    "report": report_filename,
                    "optimized_prompt": prompt_filename,
                    "summary": summary_filename
                }
            }
            
            with open(summary_filename, 'w', encoding='utf-8') as f:
                json.dump(summary_data, f, indent=2, ensure_ascii=False)
            
            # 5. Check if immediate action is needed
            action_needed = self._check_urgent_issues(analysis)
            
            # 6. Generate recommendations
            recommendations = self._generate_recommendations(analysis, improvements_summary)
            
            result = {
                "status": "success",
                "timestamp": datetime.now().isoformat(),
                "analysis_summary": {
                    "total_feedback": analysis.get("total_feedback", 0),
                    "satisfaction_rate": self._calculate_satisfaction_rate(analysis),
                    "urgent_action_needed": action_needed,
                    "improvements_identified": len(improvements_summary.get("improvements", []))
                },
                "files_generated": summary_data["files_generated"],
                "recommendations": recommendations
            }
            
            logger.info("Weekly analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error during weekly analysis: {e}")
            return {"status": "error", "message": str(e)}
    
    def _calculate_satisfaction_rate(self, analysis):
        """Calculate overall satisfaction rate"""
        feedback_dist = analysis.get("feedback_distribution", {})
        total = sum(feedback_dist.values())
        if total == 0:
            return 0
        positive = feedback_dist.get("positive", 0)
        return round((positive / total) * 100, 2)
    
    def _check_urgent_issues(self, analysis):
        """Check if there are urgent issues requiring immediate attention"""
        satisfaction_rate = self._calculate_satisfaction_rate(analysis)
        
        # Urgent if satisfaction rate is below 50%
        if satisfaction_rate < 50:
            return True
        
        # Urgent if there are many incomplete answers
        common_issues = analysis.get("common_issues", {})
        incomplete_answers = len(common_issues.get("incomplete_answer", []))
        if incomplete_answers > 5:
            return True
        
        # Urgent if recent trends show significant decline
        recent_trends = analysis.get("recent_trends", {})
        if isinstance(recent_trends, dict):
            recent_satisfaction = recent_trends.get("satisfaction_rate", 100)
            if recent_satisfaction < 40:
                return True
        
        return False
    
    def _generate_recommendations(self, analysis, improvements_summary):
        """Generate actionable recommendations"""
        recommendations = []
        
        satisfaction_rate = self._calculate_satisfaction_rate(analysis)
        common_issues = analysis.get("common_issues", {})
        
        # Satisfaction-based recommendations
        if satisfaction_rate < 70:
            recommendations.append({
                "priority": "high",
                "category": "overall_performance",
                "action": "Review and update system prompts immediately",
                "reason": f"Satisfaction rate is {satisfaction_rate}% (target: >70%)"
            })
        
        # Issue-specific recommendations
        for issue_type, cases in common_issues.items():
            if len(cases) > 2:
                priority = "high" if len(cases) > 5 else "medium"
                recommendations.append({
                    "priority": priority,
                    "category": issue_type,
                    "action": self._get_action_for_issue(issue_type),
                    "reason": f"{len(cases)} instances of {issue_type.replace('_', ' ')}"
                })
        
        return recommendations
    
    def _get_action_for_issue(self, issue_type):
        """Get recommended action for each issue type"""
        actions = {
            "incomplete_answer": "Lower similarity thresholds and improve context retrieval",
            "liquidity_issues": "Review and enhance liquidity content in knowledge base",
            "robotic_language": "Update system prompts with more natural language guidelines",
            "setup_confusion": "Add more detailed setup explanations and examples",
            "mss_problems": "Enhance MSS detection and explanation algorithms",
            "missing_context": "Improve context provision and retrieval mechanisms"
        }
        return actions.get(issue_type, "Review and address this issue type")

def main():
    """Main function to run weekly improvement analysis"""
    engine = WeeklyImprovementEngine()
    
    # Run analysis
    result = engine.run_weekly_analysis()
    
    # Print summary
    if result["status"] == "success":
        print("✅ Weekly AI Improvement Analysis Completed")
        print(f"📊 Satisfaction Rate: {result['analysis_summary']['satisfaction_rate']}%")
        print(f"🔧 Improvements Identified: {result['analysis_summary']['improvements_identified']}")
        print(f"⚠️  Urgent Action Needed: {result['analysis_summary']['urgent_action_needed']}")
        print(f"📁 Files Generated: {len(result['files_generated'])}")
        
        # Show top recommendations
        recommendations = result.get("recommendations", [])
        if recommendations:
            print("\n🎯 Top Recommendations:")
            for rec in recommendations[:3]:
                print(f"  [{rec['priority'].upper()}] {rec['action']}")
    
    else:
        print(f"❌ Analysis failed: {result.get('message', 'Unknown error')}")
    
    return result

if __name__ == "__main__":
    main() 
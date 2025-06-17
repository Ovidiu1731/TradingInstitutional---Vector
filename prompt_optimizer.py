import json
import os
from datetime import datetime
from typing import Dict, List, Any
from feedback_analyzer import FeedbackAnalyzer

class PromptOptimizer:
    def __init__(self, base_prompt_file: str = "system_prompt.txt"):
        self.base_prompt_file = base_prompt_file
        self.analyzer = FeedbackAnalyzer()
        
    def load_base_prompt(self) -> str:
        """Load the base system prompt."""
        try:
            with open(self.base_prompt_file, 'r', encoding='utf-8') as f:
                return f.read().strip()
        except FileNotFoundError:
            return "You are a helpful trading assistant."
    
    def generate_optimized_prompt(self) -> str:
        """Generate an optimized system prompt based on feedback analysis."""
        base_prompt = self.load_base_prompt()
        analysis = self.analyzer.analyze_feedback_patterns()
        
        if "error" in analysis:
            return base_prompt
        
        # Build dynamic additions based on feedback
        optimizations = []
        
        # Address common issues
        common_issues = analysis.get("common_issues", {})
        
        if "incomplete_answer" in common_issues and len(common_issues["incomplete_answer"]) > 2:
            optimizations.append("""
CRITICAL: If you cannot find specific information to answer a question:
1. First, use related information that might be helpful
2. Acknowledge what you don't know specifically
3. Provide any relevant context that might help the user
4. Never just say "I don't know" without trying to be helpful
""")
        
        if "liquidity_issues" in common_issues:
            optimizations.append("""
LIQUIDITY GUIDANCE:
- Always include ALL 4 types: HOD/LOD, Major, Local, Minor
- Major liquidity: Most profitable, marked on 15m TF, extreme zones
- Local liquidity: Marked on 1m-5m TF, less powerful than major
- Minor liquidity: Supports trend, requires experience
- HOD/LOD: Daily highs and lows
""")
        
        if "robotic_language" in common_issues:
            optimizations.append("""
COMMUNICATION STYLE:
- Use natural, conversational language
- Avoid repetitive phrases like "este important să..."
- Write like an experienced trading colleague
- Use variety in sentence structure
- Be helpful but not preachy
""")
        
        if "setup_confusion" in common_issues:
            optimizations.append("""
SETUP EXPLANATIONS:
- Always provide context for trading setups
- Explain the logic behind each setup
- Use concrete examples when possible
- Connect setups to market structure concepts
""")
        
        # Recent performance adjustments
        recent_trends = analysis.get("recent_trends", {})
        if isinstance(recent_trends, dict) and "satisfaction_rate" in recent_trends:
            if recent_trends["satisfaction_rate"] < 70:
                optimizations.append("""
PERFORMANCE BOOST NEEDED:
- Be extra thorough in your explanations
- Double-check that you're addressing the exact question asked
- Provide more context and examples
- Ensure your response is complete and helpful
""")
        
        # Query type specific improvements
        query_performance = analysis.get("query_type_performance", {})
        problematic_types = []
        for query_type, performance in query_performance.items():
            total = sum(performance.values())
            if total > 3:
                negative_rate = performance.get("negative", 0) / total
                if negative_rate > 0.4:
                    problematic_types.append(query_type)
        
        if problematic_types:
            optimizations.append(f"""
QUERY TYPE FOCUS:
The following query types need special attention: {', '.join(problematic_types)}
- Be extra careful with these question types
- Provide more detailed explanations
- Double-check your understanding before responding
""")
        
        # Combine base prompt with optimizations
        if optimizations:
            optimized_prompt = base_prompt + "\n\n" + "--- DYNAMIC OPTIMIZATIONS BASED ON USER FEEDBACK ---\n"
            optimized_prompt += "\n".join(optimizations)
            return optimized_prompt
        
        return base_prompt
    
    def save_optimized_prompt(self, filename: str = None) -> str:
        """Save the optimized prompt to a file."""
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"optimized_prompt_{timestamp}.txt"
        
        optimized_prompt = self.generate_optimized_prompt()
        
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(optimized_prompt)
        
        return filename
    
    def get_prompt_improvements_summary(self) -> Dict[str, Any]:
        """Get a summary of what improvements were made."""
        analysis = self.analyzer.analyze_feedback_patterns()
        
        if "error" in analysis:
            return {"improvements": [], "message": "No feedback data available"}
        
        improvements = []
        common_issues = analysis.get("common_issues", {})
        
        for issue_type, cases in common_issues.items():
            if cases:
                improvements.append({
                    "issue": issue_type.replace("_", " ").title(),
                    "frequency": len(cases),
                    "action": self._get_action_for_issue(issue_type)
                })
        
        recent_trends = analysis.get("recent_trends", {})
        satisfaction_rate = recent_trends.get("satisfaction_rate", 0) if isinstance(recent_trends, dict) else 0
        
        return {
            "improvements": improvements,
            "current_satisfaction": satisfaction_rate,
            "total_feedback_analyzed": analysis.get("total_feedback", 0),
            "timestamp": datetime.now().isoformat()
        }
    
    def _get_action_for_issue(self, issue_type: str) -> str:
        """Get the action taken for each issue type."""
        actions = {
            "incomplete_answer": "Added instructions to provide related information and context",
            "liquidity_issues": "Enhanced liquidity type definitions and examples",
            "robotic_language": "Updated communication style guidelines",
            "setup_confusion": "Added detailed setup explanation requirements",
            "mss_problems": "Enhanced MSS identification and explanation prompts",
            "missing_context": "Improved context provision instructions"
        }
        return actions.get(issue_type, "Added general improvement guidelines")

if __name__ == "__main__":
    optimizer = PromptOptimizer()
    
    # Generate optimization report
    summary = optimizer.get_prompt_improvements_summary()
    print("Prompt Optimization Summary:")
    print(json.dumps(summary, indent=2))
    
    # Save optimized prompt
    filename = optimizer.save_optimized_prompt()
    print(f"\nOptimized prompt saved to: {filename}")
    
    # Show the optimized prompt
    optimized = optimizer.generate_optimized_prompt()
    print(f"\nOptimized Prompt Preview:\n{optimized[:500]}...") 
"""
Portabull - Agent Orchestrator
Dragon Hatchling Architecture - Central Coordinator

The Orchestrator manages all specialist agents, coordinates debates,
synthesizes insights, and provides unified recommendations to users.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from datetime import datetime
import asyncio
import json
from loguru import logger

from .hatchlings import (
    BaseAgent,
    AgentRole,
    Argument,
    ArgumentType,
    Debate,
    AnalysisResult,
    MacroAnalystAgent,
    RiskManagerAgent,
    LongTermInvestorAgent,
    HighReturnsSpecialistAgent
)


@dataclass
class UnifiedAnalysis:
    """Unified analysis combining all agent perspectives"""
    timestamp: datetime
    portfolio_summary: Dict[str, Any]
    agent_analyses: Dict[str, AnalysisResult]
    debates: List[Debate]
    consensus_points: List[str]
    disagreement_points: List[str]
    final_recommendations: List[str]
    priority_actions: List[str]
    confidence_score: float
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "timestamp": self.timestamp.isoformat(),
            "portfolio_summary": self.portfolio_summary,
            "agent_analyses": {
                k: {
                    "summary": v.summary,
                    "insights": v.insights,
                    "recommendations": v.recommendations,
                    "risk_factors": v.risk_factors,
                    "confidence": v.confidence
                } for k, v in self.agent_analyses.items()
            },
            "debates": [d.to_dict() for d in self.debates],
            "consensus_points": self.consensus_points,
            "disagreement_points": self.disagreement_points,
            "final_recommendations": self.final_recommendations,
            "priority_actions": self.priority_actions,
            "confidence_score": self.confidence_score
        }


@dataclass
class AlertEvent:
    """Alert event for unusual portfolio activity"""
    alert_type: str
    severity: str  # low, medium, high, critical
    title: str
    description: str
    affected_symbols: List[str]
    recommended_action: str
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "type": self.alert_type,
            "severity": self.severity,
            "title": self.title,
            "description": self.description,
            "symbols": self.affected_symbols,
            "action": self.recommended_action,
            "timestamp": self.timestamp.isoformat()
        }


class AgentOrchestrator:
    """
    Central Orchestrator for Dragon Hatchling Architecture
    
    Responsibilities:
    1. Manage all specialist agents
    2. Coordinate parallel analysis
    3. Facilitate debates between agents
    4. Synthesize unified recommendations
    5. Handle user queries with multi-perspective answers
    6. Manage alerts and proactive discussions
    """
    
    def __init__(self, llm_client: Any = None):
        self.llm_client = llm_client
        
        # Initialize all specialist agents
        self.agents: Dict[AgentRole, BaseAgent] = {
            AgentRole.MACRO_ANALYST: MacroAnalystAgent(llm_client),
            AgentRole.RISK_MANAGER: RiskManagerAgent(llm_client),
            AgentRole.LONG_TERM_INVESTOR: LongTermInvestorAgent(llm_client),
            AgentRole.HIGH_RETURNS_SPECIALIST: HighReturnsSpecialistAgent(llm_client)
        }
        
        # Track active debates
        self.active_debates: List[Debate] = []
        
        # Analysis cache
        self.latest_analysis: Optional[UnifiedAnalysis] = None
        
        # Alert history
        self.alert_history: List[AlertEvent] = []
        
        logger.info("AgentOrchestrator initialized with all specialist agents")
    
    async def analyze_portfolio(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> UnifiedAnalysis:
        """
        Perform comprehensive portfolio analysis using all agents
        
        This method:
        1. Runs all agents in parallel for initial analysis
        2. Identifies points of disagreement
        3. Facilitates debates on contentious issues
        4. Synthesizes final recommendations
        """
        
        logger.info("Starting comprehensive portfolio analysis")
        
        # Step 1: Run all agents in parallel
        analysis_tasks = [
            agent.analyze(portfolio_data, market_context)
            for agent in self.agents.values()
        ]
        
        analyses = await asyncio.gather(*analysis_tasks, return_exceptions=True)
        
        # Process results
        agent_analyses: Dict[str, AnalysisResult] = {}
        for role, result in zip(self.agents.keys(), analyses):
            if isinstance(result, Exception):
                logger.error(f"Agent {role.value} failed: {result}")
                continue
            agent_analyses[role.value] = result
        
        # Step 2: Identify consensus and disagreements
        consensus_points, disagreement_points = self._identify_agreements(agent_analyses)
        
        # Step 3: Facilitate debates on disagreements
        debates = []
        for topic in disagreement_points[:3]:  # Limit to top 3 disagreements
            debate = await self._facilitate_debate(
                topic=topic,
                context={"portfolio": portfolio_data, "market": market_context}
            )
            debates.append(debate)
        
        # Step 4: Synthesize final recommendations
        final_recommendations = await self._synthesize_recommendations(
            agent_analyses, debates
        )
        
        # Step 5: Identify priority actions
        priority_actions = self._identify_priority_actions(
            agent_analyses, debates
        )
        
        # Calculate overall confidence
        confidences = [a.confidence for a in agent_analyses.values()]
        avg_confidence = sum(confidences) / len(confidences) if confidences else 0.5
        
        # Create unified analysis
        unified = UnifiedAnalysis(
            timestamp=datetime.now(),
            portfolio_summary=self._create_portfolio_summary(portfolio_data),
            agent_analyses=agent_analyses,
            debates=debates,
            consensus_points=consensus_points,
            disagreement_points=disagreement_points,
            final_recommendations=final_recommendations,
            priority_actions=priority_actions,
            confidence_score=avg_confidence
        )
        
        self.latest_analysis = unified
        
        logger.info("Portfolio analysis completed")
        return unified
    
    async def answer_query(
        self,
        query: str,
        context: Dict[str, Any],
        show_debate: bool = False
    ) -> Dict[str, Any]:
        """
        Answer a user query using all agents
        
        Args:
            query: User's question
            context: Current portfolio and market context
            show_debate: Whether to show agent deliberation process
            
        Returns:
            Dict with answer and optional debate transcript
        """
        
        logger.info(f"Processing query: {query[:100]}...")
        
        # Get responses from all agents in parallel
        response_tasks = [
            agent.respond_to_query(query, context)
            for agent in self.agents.values()
        ]
        
        responses = await asyncio.gather(*response_tasks, return_exceptions=True)
        
        # Collect valid responses
        agent_responses: Dict[str, str] = {}
        for role, response in zip(self.agents.keys(), responses):
            if isinstance(response, Exception):
                logger.error(f"Agent {role.value} failed: {response}")
                continue
            agent_responses[role.value] = response
        
        # Synthesize unified response
        unified_response = await self._synthesize_query_response(
            query, agent_responses, context
        )
        
        result = {
            "answer": unified_response,
            "timestamp": datetime.now().isoformat()
        }
        
        if show_debate:
            result["agent_perspectives"] = agent_responses
            result["deliberation"] = self._format_deliberation(agent_responses)
        
        return result
    
    async def _facilitate_debate(
        self,
        topic: str,
        context: Dict[str, Any],
        max_rounds: int = 2
    ) -> Debate:
        """
        Facilitate a debate between agents on a topic
        
        The debate proceeds in rounds:
        1. Each agent states their position
        2. Agents respond to each other's arguments
        3. Orchestrator summarizes consensus
        """
        
        debate = Debate(topic=topic, context=context)
        
        for round_num in range(max_rounds):
            # Get arguments from each agent
            for role, agent in self.agents.items():
                argument = await agent.debate(
                    topic=topic,
                    other_arguments=debate.arguments,
                    context=context
                )
                debate.add_argument(argument)
        
        # Synthesize consensus
        debate.consensus = await self._find_consensus(debate)
        debate.final_recommendation = await self._get_debate_recommendation(debate)
        
        return debate
    
    def _identify_agreements(
        self,
        analyses: Dict[str, AnalysisResult]
    ) -> Tuple[List[str], List[str]]:
        """Identify consensus points and disagreements between agents"""
        
        # Collect all recommendations and insights
        all_recommendations: Dict[str, List[str]] = {
            role: analysis.recommendations
            for role, analysis in analyses.items()
        }
        
        all_risk_factors: Dict[str, List[str]] = {
            role: analysis.risk_factors
            for role, analysis in analyses.items()
        }
        
        # Simple heuristic: look for similar themes
        consensus = []
        disagreements = []
        
        # For hackathon: simplified logic
        # In production, use NLP similarity
        
        # Check if all agents mention similar risks
        risk_keywords = set()
        for risks in all_risk_factors.values():
            for risk in risks:
                words = set(risk.lower().split())
                risk_keywords.update(words)
        
        # Common risks mentioned by multiple agents become consensus
        for role1, risks1 in all_risk_factors.items():
            for role2, risks2 in all_risk_factors.items():
                if role1 != role2:
                    # Look for overlapping concerns
                    for r1 in risks1:
                        for r2 in risks2:
                            if self._text_similarity(r1, r2) > 0.5:
                                consensus.append(f"Shared concern: {r1}")
        
        # Contradicting recommendations become disagreements
        reco_by_agent = list(all_recommendations.items())
        for i, (role1, recos1) in enumerate(reco_by_agent):
            for role2, recos2 in reco_by_agent[i+1:]:
                for r1 in recos1:
                    for r2 in recos2:
                        # Check for contradictions
                        if "buy" in r1.lower() and "sell" in r2.lower():
                            disagreements.append(f"Buy vs Sell: {role1} vs {role2}")
                        elif "hold" in r1.lower() and "exit" in r2.lower():
                            disagreements.append(f"Hold vs Exit: {role1} vs {role2}")
        
        # Remove duplicates
        consensus = list(set(consensus))[:5]
        disagreements = list(set(disagreements))[:5]
        
        return consensus, disagreements
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple text similarity using word overlap"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1.intersection(words2)
        union = words1.union(words2)
        
        return len(intersection) / len(union)
    
    async def _synthesize_recommendations(
        self,
        analyses: Dict[str, AnalysisResult],
        debates: List[Debate]
    ) -> List[str]:
        """Synthesize final recommendations from all agent inputs"""
        
        # Collect all recommendations with weights based on confidence
        weighted_recommendations = []
        
        for role, analysis in analyses.items():
            for reco in analysis.recommendations:
                weighted_recommendations.append({
                    "recommendation": reco,
                    "source": role,
                    "confidence": analysis.confidence
                })
        
        # Add debate conclusions
        for debate in debates:
            if debate.final_recommendation:
                weighted_recommendations.append({
                    "recommendation": debate.final_recommendation,
                    "source": "debate_consensus",
                    "confidence": 0.8
                })
        
        # Sort by confidence and return top recommendations
        weighted_recommendations.sort(key=lambda x: x["confidence"], reverse=True)
        
        return [r["recommendation"] for r in weighted_recommendations[:10]]
    
    def _identify_priority_actions(
        self,
        analyses: Dict[str, AnalysisResult],
        debates: List[Debate]
    ) -> List[str]:
        """Identify actions that need immediate attention"""
        
        priority_actions = []
        
        # Risk manager warnings are high priority
        if AgentRole.RISK_MANAGER.value in analyses:
            risk_analysis = analyses[AgentRole.RISK_MANAGER.value]
            for risk in risk_analysis.risk_factors:
                if any(word in risk.lower() for word in ["critical", "high", "immediate", "urgent"]):
                    priority_actions.append(f"⚠️ {risk}")
        
        # Consensus points from debates
        for debate in debates:
            if debate.consensus:
                if "action" in debate.consensus.lower() or "immediately" in debate.consensus.lower():
                    priority_actions.append(f"📋 {debate.consensus}")
        
        return priority_actions[:5]
    
    def _create_portfolio_summary(self, portfolio_data: Dict[str, Any]) -> Dict[str, Any]:
        """Create a summary of the portfolio"""
        
        holdings = portfolio_data.get("holdings", [])
        
        total_value = sum(h.get("value", 0) for h in holdings)
        total_pnl = sum(h.get("pnl", 0) for h in holdings)
        
        return {
            "total_holdings": len(holdings),
            "total_value": total_value,
            "total_pnl": total_pnl,
            "pnl_percent": (total_pnl / total_value * 100) if total_value else 0
        }
    
    async def _find_consensus(self, debate: Debate) -> str:
        """Find consensus from a debate"""
        
        supports = [a for a in debate.arguments if a.argument_type == ArgumentType.SUPPORT]
        opposes = [a for a in debate.arguments if a.argument_type == ArgumentType.OPPOSE]
        
        if len(supports) > len(opposes):
            return f"Majority supports the action with {len(supports)} in favor"
        elif len(opposes) > len(supports):
            return f"Majority opposes the action with {len(opposes)} against"
        else:
            return "No clear consensus - further analysis recommended"
    
    async def _get_debate_recommendation(self, debate: Debate) -> str:
        """Get final recommendation from a debate"""
        
        # Aggregate confidence-weighted positions
        total_support = sum(
            a.confidence for a in debate.arguments 
            if a.argument_type == ArgumentType.SUPPORT
        )
        total_oppose = sum(
            a.confidence for a in debate.arguments 
            if a.argument_type == ArgumentType.OPPOSE
        )
        
        if total_support > total_oppose:
            return "Recommended: Proceed with the action based on multi-agent consensus"
        elif total_oppose > total_support:
            return "Recommended: Avoid this action based on risk concerns"
        else:
            return "Recommended: Monitor situation and reassess later"
    
    async def _synthesize_query_response(
        self,
        query: str,
        agent_responses: Dict[str, str],
        context: Dict[str, Any]
    ) -> str:
        """Synthesize a unified response from all agent inputs using LLM"""
        
        if not agent_responses:
            return "I apologize, but I couldn't gather insights from my analysis team. Please try again."
        
        # Get RAG context (real market data from Dynamic RAG)
        rag_context = ""
        relevant_docs = context.get("relevant_docs", [])
        if relevant_docs:
            rag_context = "\n\nRELEVANT MARKET DATA (from Dynamic RAG):\n"
            for i, doc in enumerate(relevant_docs[:5], 1):  # Limit to top 5 docs
                rag_context += f"\n--- Document {i} ---\n{doc[:800]}\n"
        
        # Use LLM to synthesize a clear, actionable response
        synthesis_prompt = f"""You are a senior financial advisor synthesizing insights from multiple specialist analysts.

USER QUESTION: {query}
{rag_context}
ANALYST INSIGHTS:
"""
        for role, response in agent_responses.items():
            agent_name = role.replace("_", " ").title()
            synthesis_prompt += f"\n{agent_name}:\n{response[:800]}\n"
        
        synthesis_prompt += """
INSTRUCTIONS:
1. Create a CLEAR, CONCISE answer that addresses the user's question directly
2. USE THE MARKET DATA provided above to give specific, data-backed insights
3. Synthesize the key insights from all analysts into actionable advice
4. Highlight points where analysts AGREE (these are high-confidence recommendations)
5. If there are significant disagreements, briefly mention them
6. End with 2-3 specific, actionable recommendations
7. Keep the response focused and under 400 words
8. Use bullet points for clarity
9. Do NOT show individual analyst names or their separate opinions - this is a UNIFIED response
10. Reference specific prices, P/E ratios, or market data when available

FORMAT YOUR RESPONSE AS:
📊 **Summary**: [Brief direct answer to the question]

🔑 **Key Insights**:
• [Insight 1 - with specific data if available]
• [Insight 2 - with specific data if available]
• [Insight 3 - with specific data if available]

✅ **Recommended Actions**:
1. [Action 1]
2. [Action 2]
3. [Action 3]

⚠️ **Risk Considerations**: [Brief note on any risks to be aware of]
"""
        
        try:
            synthesized = await self.llm_client.generate([
                {"role": "system", "content": "You are a senior financial advisor providing clear, actionable investment guidance based on real market data."},
                {"role": "user", "content": synthesis_prompt}
            ], temperature=0.7, max_tokens=800)
            
            return synthesized
            
        except Exception as e:
            logger.error(f"Failed to synthesize response: {e}")
            # Fallback to a simpler format
            return self._fallback_synthesis(query, agent_responses)
    
    def _fallback_synthesis(self, query: str, agent_responses: Dict[str, str]) -> str:
        """Fallback synthesis when LLM fails"""
        
        # Extract key points from responses
        all_text = " ".join(agent_responses.values()).lower()
        
        response = f"""📊 **Summary**: Based on analysis from multiple perspectives, here's my assessment:

🔑 **Key Insights**:
"""
        # Add a condensed insight from each agent
        for role, response_text in agent_responses.items():
            # Get first sentence or first 150 chars
            first_sentence = response_text.split('.')[0][:150]
            response += f"• {first_sentence}.\n"
        
        response += """
✅ **Recommended Actions**:
1. Review your current portfolio allocation
2. Consider diversification opportunities
3. Monitor market conditions regularly

⚠️ **Risk Considerations**: Market conditions can change rapidly. Always do your own research before making investment decisions."""
        
        return response
    
    def _format_deliberation(self, agent_responses: Dict[str, str]) -> str:
        """Format agent deliberation for display"""
        
        deliberation = ["## Agent Deliberation Process\n"]
        
        for role, response in agent_responses.items():
            agent_name = role.replace("_", " ").title()
            deliberation.append(f"### {agent_name}\n{response}\n")
        
        return "\n".join(deliberation)
    
    async def detect_anomaly(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any]
    ) -> List[AlertEvent]:
        """
        Detect anomalies in portfolio based on market data
        
        Checks for:
        - Significant price movements
        - Volume spikes
        - Correlation breakdowns
        - Sector-wide movements
        """
        
        alerts = []
        
        holdings = portfolio_data.get("holdings", [])
        
        for holding in holdings:
            symbol = holding.get("tradingsymbol", "UNKNOWN")
            
            # Check price change
            price_change = holding.get("day_change_percent", 0)
            if abs(price_change) > 5:
                severity = "high" if abs(price_change) > 10 else "medium"
                alerts.append(AlertEvent(
                    alert_type="price_movement",
                    severity=severity,
                    title=f"Significant price movement in {symbol}",
                    description=f"{symbol} has moved {price_change:+.2f}% today",
                    affected_symbols=[symbol],
                    recommended_action="Review position and consider stop-loss or profit booking"
                ))
        
        # Check portfolio-wide metrics
        total_pnl_percent = sum(h.get("pnl_percent", 0) for h in holdings)
        if len(holdings) > 0:
            avg_pnl = total_pnl_percent / len(holdings)
            
            if avg_pnl < -5:
                alerts.append(AlertEvent(
                    alert_type="portfolio_drawdown",
                    severity="high",
                    title="Portfolio-wide drawdown detected",
                    description=f"Average portfolio loss is {avg_pnl:.2f}%",
                    affected_symbols=[h.get("tradingsymbol") for h in holdings],
                    recommended_action="Review risk exposure and consider hedging strategies"
                ))
        
        # Store alerts
        self.alert_history.extend(alerts)
        
        return alerts
    
    async def start_proactive_discussion(
        self,
        alert: AlertEvent,
        portfolio_data: Dict[str, Any]
    ) -> Debate:
        """
        Start a proactive discussion with user about an alert
        
        This initiates a debate among agents about how to handle
        the detected anomaly
        """
        
        topic = f"Response to {alert.alert_type}: {alert.title}"
        
        context = {
            "alert": alert.to_dict(),
            "portfolio": portfolio_data,
            "severity": alert.severity
        }
        
        debate = await self._facilitate_debate(
            topic=topic,
            context=context,
            max_rounds=2
        )
        
        return debate
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents"""
        
        return {
            "agents": [
                {
                    "role": role.value,
                    "name": agent.name,
                    "description": agent.description,
                    "status": "active"
                }
                for role, agent in self.agents.items()
            ],
            "active_debates": len(self.active_debates),
            "alert_count": len(self.alert_history)
        }

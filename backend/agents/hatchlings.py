"""
Portabull - Dragon Hatchling Architecture
Multi-Agent System for Portfolio Analysis

The Dragon Hatchling architecture implements a collaborative AI system
where multiple specialized agents (hatchlings) work together to provide
comprehensive portfolio analysis. Each agent represents a different
investment perspective and they "argue" to reach the best recommendation.

Agents:
- Macro Analyst: Macroeconomic perspective
- Risk Manager: Risk assessment and mitigation
- Long-term Investor: Value investing perspective
- High Returns Specialist: Growth and momentum focus

The agents debate and the Orchestrator synthesizes their insights
into actionable recommendations.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime
import asyncio
import json
from loguru import logger


class AgentRole(Enum):
    """Enumeration of agent roles in the Dragon Hatchling architecture"""
    MACRO_ANALYST = "macro_analyst"
    RISK_MANAGER = "risk_manager"
    LONG_TERM_INVESTOR = "long_term_investor"
    HIGH_RETURNS_SPECIALIST = "high_returns_specialist"
    ORCHESTRATOR = "orchestrator"


class ArgumentType(Enum):
    """Types of arguments in agent debates"""
    SUPPORT = "support"
    OPPOSE = "oppose"
    NEUTRAL = "neutral"
    CLARIFY = "clarify"


@dataclass
class Argument:
    """Represents an argument made by an agent"""
    agent_role: AgentRole
    content: str
    argument_type: ArgumentType
    confidence: float  # 0.0 to 1.0
    supporting_data: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "agent": self.agent_role.value,
            "content": self.content,
            "type": self.argument_type.value,
            "confidence": self.confidence,
            "data": self.supporting_data,
            "timestamp": self.timestamp.isoformat()
        }


@dataclass
class Debate:
    """Represents a debate between agents on a topic"""
    topic: str
    context: Dict[str, Any]
    arguments: List[Argument] = field(default_factory=list)
    consensus: Optional[str] = None
    final_recommendation: Optional[str] = None
    created_at: datetime = field(default_factory=datetime.now)
    
    def add_argument(self, argument: Argument):
        self.arguments.append(argument)
    
    def get_arguments_by_agent(self, role: AgentRole) -> List[Argument]:
        return [a for a in self.arguments if a.agent_role == role]
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "topic": self.topic,
            "context": self.context,
            "arguments": [a.to_dict() for a in self.arguments],
            "consensus": self.consensus,
            "recommendation": self.final_recommendation,
            "created_at": self.created_at.isoformat()
        }


@dataclass
class AnalysisResult:
    """Result of an agent's analysis"""
    agent_role: AgentRole
    summary: str
    insights: List[str]
    recommendations: List[str]
    risk_factors: List[str]
    confidence: float
    raw_data: Dict[str, Any] = field(default_factory=dict)


class BaseAgent(ABC):
    """
    Base class for all Dragon Hatchling agents
    
    Each agent specializes in a different aspect of portfolio analysis
    and can participate in debates with other agents.
    """
    
    def __init__(
        self,
        role: AgentRole,
        name: str,
        description: str,
        system_prompt: str,
        llm_client: Any = None
    ):
        self.role = role
        self.name = name
        self.description = description
        self.system_prompt = system_prompt
        self.llm_client = llm_client
        self.conversation_history: List[Dict[str, str]] = []
    
    @abstractmethod
    async def analyze(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze portfolio from this agent's perspective"""
        pass
    
    @abstractmethod
    async def debate(
        self,
        topic: str,
        other_arguments: List[Argument],
        context: Dict[str, Any]
    ) -> Argument:
        """Participate in a debate with other agents"""
        pass
    
    @abstractmethod
    async def respond_to_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Respond to a user query from this agent's perspective"""
        pass
    
    async def _call_llm(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7
    ) -> str:
        """Call the LLM with the given messages"""
        if self.llm_client is None:
            raise ValueError("LLM client not initialized")
        
        return await self.llm_client.generate(
            messages=messages,
            temperature=temperature
        )
    
    def _build_messages(
        self,
        user_message: str,
        context: Dict[str, Any] = None
    ) -> List[Dict[str, str]]:
        """Build message list for LLM call"""
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add RAG context if provided (real market data)
        if context and context.get("relevant_docs"):
            rag_content = "Real-time Market Data from Dynamic RAG:\n"
            for i, doc in enumerate(context["relevant_docs"][:3], 1):  # Top 3 docs per agent
                rag_content += f"\n--- Data {i} ---\n{doc[:600]}\n"
            messages.append({
                "role": "system",
                "content": rag_content
            })
        
        # Add portfolio context if provided
        if context:
            # Create a focused context without the large relevant_docs
            focused_context = {k: v for k, v in context.items() if k != "relevant_docs"}
            if focused_context:
                context_str = json.dumps(focused_context, indent=2, default=str)
                messages.append({
                    "role": "system",
                    "content": f"Portfolio context:\n{context_str}"
                })
        
        # Add conversation history
        messages.extend(self.conversation_history[-10:])  # Last 10 messages
        
        # Add current message
        messages.append({"role": "user", "content": user_message})
        
        return messages


class MacroAnalystAgent(BaseAgent):
    """
    Macro Analyst Agent
    
    Focuses on:
    - Macroeconomic indicators
    - Sector rotation
    - Market cycles
    - Global economic trends
    """
    
    def __init__(self, llm_client: Any = None):
        super().__init__(
            role=AgentRole.MACRO_ANALYST,
            name="Macro Analyst",
            description="Expert in macroeconomic analysis and market cycles",
            system_prompt="""You are an expert Macro Analyst AI. Your role is to:
- Analyze macroeconomic indicators (GDP, inflation, interest rates, employment)
- Evaluate sector rotations and market cycles
- Assess global economic trends and their impact on portfolios
- Provide insights on monetary and fiscal policies
- Consider geopolitical factors affecting markets

When analyzing a portfolio, focus on how macro factors might affect the holdings.
Be specific with data and provide actionable insights.
Always support your analysis with current economic data when available.""",
            llm_client=llm_client
        )
    
    async def analyze(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze portfolio from macro perspective"""
        
        prompt = f"""Analyze the following portfolio from a macroeconomic perspective:

Portfolio:
{json.dumps(portfolio_data, indent=2, default=str)}

Market Context:
{json.dumps(market_context, indent=2, default=str)}

Provide:
1. Summary of macro implications
2. Key insights about sector exposure
3. Recommendations based on current economic cycle
4. Risk factors from macro perspective
5. Your confidence level (0-100%)

Format your response as JSON with keys: summary, insights (list), recommendations (list), risk_factors (list), confidence (number)"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return AnalysisResult(
                agent_role=self.role,
                summary=result.get("summary", ""),
                insights=result.get("insights", []),
                recommendations=result.get("recommendations", []),
                risk_factors=result.get("risk_factors", []),
                confidence=result.get("confidence", 50) / 100
            )
        except json.JSONDecodeError:
            # Parse as text if not valid JSON
            return AnalysisResult(
                agent_role=self.role,
                summary=response,
                insights=[],
                recommendations=[],
                risk_factors=[],
                confidence=0.5
            )
    
    async def debate(
        self,
        topic: str,
        other_arguments: List[Argument],
        context: Dict[str, Any]
    ) -> Argument:
        """Participate in debate from macro perspective"""
        
        other_args_str = "\n".join([
            f"- {a.agent_role.value}: {a.content}" 
            for a in other_arguments
        ])
        
        prompt = f"""Topic: {topic}

Other agents' arguments:
{other_args_str}

Context:
{json.dumps(context, indent=2, default=str)}

As the Macro Analyst, provide your argument on this topic.
Consider macroeconomic factors and how they relate to the discussion.
State whether you SUPPORT, OPPOSE, or are NEUTRAL on the main point.
Rate your confidence (0-100%).

Format: 
TYPE: [SUPPORT/OPPOSE/NEUTRAL]
CONFIDENCE: [0-100]
ARGUMENT: [Your detailed argument]"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        # Parse response
        arg_type = ArgumentType.NEUTRAL
        confidence = 0.5
        content = response
        
        if "SUPPORT" in response.upper()[:100]:
            arg_type = ArgumentType.SUPPORT
        elif "OPPOSE" in response.upper()[:100]:
            arg_type = ArgumentType.OPPOSE
        
        return Argument(
            agent_role=self.role,
            content=content,
            argument_type=arg_type,
            confidence=confidence,
            supporting_data=context
        )
    
    async def respond_to_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Respond to user query from macro perspective"""
        
        prompt = f"""User Query: {query}

Context:
{json.dumps(context, indent=2, default=str)}

Provide a response from your perspective as a Macro Analyst.
Focus on macroeconomic factors relevant to the query."""
        
        messages = self._build_messages(prompt)
        return await self._call_llm(messages)


class RiskManagerAgent(BaseAgent):
    """
    Risk Manager Agent
    
    Focuses on:
    - Portfolio risk metrics
    - Concentration risk
    - Downside protection
    - Hedging strategies
    """
    
    def __init__(self, llm_client: Any = None):
        super().__init__(
            role=AgentRole.RISK_MANAGER,
            name="Risk Manager",
            description="Expert in risk assessment and portfolio protection",
            system_prompt="""You are an expert Risk Manager AI. Your role is to:
- Calculate and analyze portfolio risk metrics (Beta, Sharpe Ratio, VaR, Max Drawdown)
- Identify concentration risks and correlation issues
- Suggest hedging strategies and risk mitigation
- Monitor position sizing and leverage
- Alert on excessive exposure to specific sectors/stocks

When analyzing a portfolio, focus on potential risks and how to protect capital.
Use quantitative risk measures and be conservative in your assessments.
Always prioritize capital preservation.""",
            llm_client=llm_client
        )
    
    async def analyze(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze portfolio from risk perspective"""
        
        prompt = f"""Analyze the following portfolio from a risk management perspective:

Portfolio:
{json.dumps(portfolio_data, indent=2, default=str)}

Market Context:
{json.dumps(market_context, indent=2, default=str)}

Provide:
1. Summary of risk assessment
2. Key risk factors identified
3. Recommendations for risk mitigation
4. Critical warnings (if any)
5. Your confidence level (0-100%)

Format your response as JSON with keys: summary, insights (list), recommendations (list), risk_factors (list), confidence (number)"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return AnalysisResult(
                agent_role=self.role,
                summary=result.get("summary", ""),
                insights=result.get("insights", []),
                recommendations=result.get("recommendations", []),
                risk_factors=result.get("risk_factors", []),
                confidence=result.get("confidence", 50) / 100
            )
        except json.JSONDecodeError:
            return AnalysisResult(
                agent_role=self.role,
                summary=response,
                insights=[],
                recommendations=[],
                risk_factors=[],
                confidence=0.5
            )
    
    async def debate(
        self,
        topic: str,
        other_arguments: List[Argument],
        context: Dict[str, Any]
    ) -> Argument:
        """Participate in debate from risk perspective"""
        
        other_args_str = "\n".join([
            f"- {a.agent_role.value}: {a.content}" 
            for a in other_arguments
        ])
        
        prompt = f"""Topic: {topic}

Other agents' arguments:
{other_args_str}

Context:
{json.dumps(context, indent=2, default=str)}

As the Risk Manager, provide your argument on this topic.
Focus on risk implications and capital protection.
State whether you SUPPORT, OPPOSE, or are NEUTRAL on the main point.
Rate your confidence (0-100%).

Format: 
TYPE: [SUPPORT/OPPOSE/NEUTRAL]
CONFIDENCE: [0-100]
ARGUMENT: [Your detailed argument]"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        arg_type = ArgumentType.NEUTRAL
        if "SUPPORT" in response.upper()[:100]:
            arg_type = ArgumentType.SUPPORT
        elif "OPPOSE" in response.upper()[:100]:
            arg_type = ArgumentType.OPPOSE
        
        return Argument(
            agent_role=self.role,
            content=response,
            argument_type=arg_type,
            confidence=0.5,
            supporting_data=context
        )
    
    async def respond_to_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Respond to user query from risk perspective"""
        
        prompt = f"""User Query: {query}

Context:
{json.dumps(context, indent=2, default=str)}

Provide a response from your perspective as a Risk Manager.
Focus on risk factors and protective measures relevant to the query."""
        
        messages = self._build_messages(prompt)
        return await self._call_llm(messages)


class LongTermInvestorAgent(BaseAgent):
    """
    Long-term Investor Agent
    
    Focuses on:
    - Fundamental analysis
    - Value investing
    - Business quality
    - Wealth building
    """
    
    def __init__(self, llm_client: Any = None):
        super().__init__(
            role=AgentRole.LONG_TERM_INVESTOR,
            name="Long-term Investor",
            description="Expert in value investing and fundamental analysis",
            system_prompt="""You are an expert Long-term Investor AI (like Warren Buffett). Your role is to:
- Analyze company fundamentals (PE, PB, ROE, Debt/Equity, FCF)
- Evaluate competitive advantages (moats)
- Assess management quality and corporate governance
- Focus on intrinsic value and margin of safety
- Recommend buy-and-hold strategies for wealth creation

When analyzing a portfolio, focus on long-term value creation potential.
Ignore short-term volatility and focus on business quality.
Think in terms of decades, not days.""",
            llm_client=llm_client
        )
    
    async def analyze(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze portfolio from long-term value perspective"""
        
        prompt = f"""Analyze the following portfolio from a long-term value investing perspective:

Portfolio:
{json.dumps(portfolio_data, indent=2, default=str)}

Market Context:
{json.dumps(market_context, indent=2, default=str)}

Provide:
1. Summary of long-term value assessment
2. Key insights about business quality
3. Recommendations for long-term wealth building
4. Concerns about any holdings
5. Your confidence level (0-100%)

Format your response as JSON with keys: summary, insights (list), recommendations (list), risk_factors (list), confidence (number)"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return AnalysisResult(
                agent_role=self.role,
                summary=result.get("summary", ""),
                insights=result.get("insights", []),
                recommendations=result.get("recommendations", []),
                risk_factors=result.get("risk_factors", []),
                confidence=result.get("confidence", 50) / 100
            )
        except json.JSONDecodeError:
            return AnalysisResult(
                agent_role=self.role,
                summary=response,
                insights=[],
                recommendations=[],
                risk_factors=[],
                confidence=0.5
            )
    
    async def debate(
        self,
        topic: str,
        other_arguments: List[Argument],
        context: Dict[str, Any]
    ) -> Argument:
        """Participate in debate from long-term value perspective"""
        
        other_args_str = "\n".join([
            f"- {a.agent_role.value}: {a.content}" 
            for a in other_arguments
        ])
        
        prompt = f"""Topic: {topic}

Other agents' arguments:
{other_args_str}

Context:
{json.dumps(context, indent=2, default=str)}

As the Long-term Investor, provide your argument on this topic.
Focus on fundamental value and long-term wealth creation.
State whether you SUPPORT, OPPOSE, or are NEUTRAL on the main point.
Rate your confidence (0-100%).

Format: 
TYPE: [SUPPORT/OPPOSE/NEUTRAL]
CONFIDENCE: [0-100]
ARGUMENT: [Your detailed argument]"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        arg_type = ArgumentType.NEUTRAL
        if "SUPPORT" in response.upper()[:100]:
            arg_type = ArgumentType.SUPPORT
        elif "OPPOSE" in response.upper()[:100]:
            arg_type = ArgumentType.OPPOSE
        
        return Argument(
            agent_role=self.role,
            content=response,
            argument_type=arg_type,
            confidence=0.5,
            supporting_data=context
        )
    
    async def respond_to_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Respond to user query from long-term perspective"""
        
        prompt = f"""User Query: {query}

Context:
{json.dumps(context, indent=2, default=str)}

Provide a response from your perspective as a Long-term Value Investor.
Focus on fundamentals and long-term wealth creation."""
        
        messages = self._build_messages(prompt)
        return await self._call_llm(messages)


class HighReturnsSpecialistAgent(BaseAgent):
    """
    High Returns Specialist Agent
    
    Focuses on:
    - Growth opportunities
    - Momentum strategies
    - Alpha generation
    - Tactical allocation
    """
    
    def __init__(self, llm_client: Any = None):
        super().__init__(
            role=AgentRole.HIGH_RETURNS_SPECIALIST,
            name="High Returns Specialist",
            description="Expert in growth investing and alpha generation",
            system_prompt="""You are an expert High Returns Specialist AI. Your role is to:
- Identify high-growth opportunities and emerging trends
- Analyze momentum indicators and technical patterns
- Spot potential multibaggers and turnaround stories
- Evaluate risk-reward ratios for aggressive positions
- Suggest tactical allocation shifts for higher returns

When analyzing a portfolio, focus on return enhancement opportunities.
Balance aggression with calculated risk-taking.
Look for asymmetric risk-reward opportunities.""",
            llm_client=llm_client
        )
    
    async def analyze(
        self,
        portfolio_data: Dict[str, Any],
        market_context: Dict[str, Any]
    ) -> AnalysisResult:
        """Analyze portfolio for return enhancement opportunities"""
        
        prompt = f"""Analyze the following portfolio from a high returns perspective:

Portfolio:
{json.dumps(portfolio_data, indent=2, default=str)}

Market Context:
{json.dumps(market_context, indent=2, default=str)}

Provide:
1. Summary of return enhancement opportunities
2. Key insights about growth potential
3. Recommendations for higher returns
4. Risk factors to consider
5. Your confidence level (0-100%)

Format your response as JSON with keys: summary, insights (list), recommendations (list), risk_factors (list), confidence (number)"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        try:
            result = json.loads(response)
            return AnalysisResult(
                agent_role=self.role,
                summary=result.get("summary", ""),
                insights=result.get("insights", []),
                recommendations=result.get("recommendations", []),
                risk_factors=result.get("risk_factors", []),
                confidence=result.get("confidence", 50) / 100
            )
        except json.JSONDecodeError:
            return AnalysisResult(
                agent_role=self.role,
                summary=response,
                insights=[],
                recommendations=[],
                risk_factors=[],
                confidence=0.5
            )
    
    async def debate(
        self,
        topic: str,
        other_arguments: List[Argument],
        context: Dict[str, Any]
    ) -> Argument:
        """Participate in debate from high returns perspective"""
        
        other_args_str = "\n".join([
            f"- {a.agent_role.value}: {a.content}" 
            for a in other_arguments
        ])
        
        prompt = f"""Topic: {topic}

Other agents' arguments:
{other_args_str}

Context:
{json.dumps(context, indent=2, default=str)}

As the High Returns Specialist, provide your argument on this topic.
Focus on return potential and alpha generation.
State whether you SUPPORT, OPPOSE, or are NEUTRAL on the main point.
Rate your confidence (0-100%).

Format: 
TYPE: [SUPPORT/OPPOSE/NEUTRAL]
CONFIDENCE: [0-100]
ARGUMENT: [Your detailed argument]"""
        
        messages = self._build_messages(prompt)
        response = await self._call_llm(messages)
        
        arg_type = ArgumentType.NEUTRAL
        if "SUPPORT" in response.upper()[:100]:
            arg_type = ArgumentType.SUPPORT
        elif "OPPOSE" in response.upper()[:100]:
            arg_type = ArgumentType.OPPOSE
        
        return Argument(
            agent_role=self.role,
            content=response,
            argument_type=arg_type,
            confidence=0.5,
            supporting_data=context
        )
    
    async def respond_to_query(
        self,
        query: str,
        context: Dict[str, Any]
    ) -> str:
        """Respond to user query from high returns perspective"""
        
        prompt = f"""User Query: {query}

Context:
{json.dumps(context, indent=2, default=str)}

Provide a response from your perspective as a High Returns Specialist.
Focus on growth and return enhancement opportunities."""
        
        messages = self._build_messages(prompt)
        return await self._call_llm(messages)

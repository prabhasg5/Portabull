"""
Portabull - Hypothesis Generator
Automated investment hypothesis generation from portfolio analysis

This module generates intelligent investment hypotheses by:
1. Analyzing portfolio composition and performance
2. Identifying patterns and correlations
3. Using LLM to generate actionable hypotheses
4. Continuously updating hypotheses as market conditions change
"""

import asyncio
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, field
from datetime import datetime
from loguru import logger
import json


@dataclass
class InvestmentHypothesis:
    """Represents an investment hypothesis"""
    hypothesis_id: str
    title: str
    description: str
    hypothesis_type: str  # bullish, bearish, neutral, hedging, rebalancing
    affected_symbols: List[str]
    confidence: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    potential_actions: List[str]
    risk_factors: List[str]
    time_horizon: str  # short-term, medium-term, long-term
    generated_at: datetime = field(default_factory=datetime.now)
    status: str = "active"  # active, validated, invalidated, expired
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "title": self.title,
            "description": self.description,
            "hypothesis_type": self.hypothesis_type,
            "affected_symbols": self.affected_symbols,
            "confidence": self.confidence,
            "supporting_evidence": self.supporting_evidence,
            "potential_actions": self.potential_actions,
            "risk_factors": self.risk_factors,
            "time_horizon": self.time_horizon,
            "generated_at": self.generated_at.isoformat(),
            "status": self.status
        }


class HypothesisGenerator:
    """
    Generates investment hypotheses automatically from portfolio data
    
    Hypothesis Types:
    1. CONCENTRATION: Portfolio too concentrated in sector/stock
    2. MOMENTUM: Stocks showing strong momentum
    3. REVERSION: Oversold/overbought conditions
    4. CORRELATION: Hidden correlations between holdings
    5. MACRO: Macro-economic impacts on portfolio
    6. RISK: Risk-based hypotheses (volatility, drawdown)
    7. OPPORTUNITY: New investment opportunities
    """
    
    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self._hypotheses: Dict[str, InvestmentHypothesis] = {}
        self._hypothesis_counter = 0
    
    async def generate_hypotheses(
        self,
        portfolio_data: Dict[str, Any],
        market_data: Dict[str, Any] = None
    ) -> List[InvestmentHypothesis]:
        """
        Generate hypotheses from portfolio and market data
        """
        
        logger.info("Generating investment hypotheses...")
        hypotheses = []
        
        holdings = portfolio_data.get("holdings", [])
        if not holdings:
            return hypotheses
        
        # 1. Concentration Analysis
        concentration_hyp = await self._analyze_concentration(holdings)
        if concentration_hyp:
            hypotheses.append(concentration_hyp)
        
        # 2. Performance Analysis
        performance_hyps = await self._analyze_performance(holdings)
        hypotheses.extend(performance_hyps)
        
        # 3. Sector Analysis
        sector_hyps = await self._analyze_sectors(holdings)
        hypotheses.extend(sector_hyps)
        
        # 4. Risk Analysis
        risk_hyps = await self._analyze_risk(holdings, market_data)
        hypotheses.extend(risk_hyps)
        
        # 5. Opportunity Analysis
        opportunity_hyps = await self._find_opportunities(holdings, market_data)
        hypotheses.extend(opportunity_hyps)
        
        # Store hypotheses
        for hyp in hypotheses:
            self._hypotheses[hyp.hypothesis_id] = hyp
        
        # Use LLM to enhance hypotheses if available
        if self.llm_client and hypotheses:
            hypotheses = await self._enhance_with_llm(hypotheses, portfolio_data)
        
        logger.info(f"Generated {len(hypotheses)} hypotheses")
        return hypotheses
    
    async def _analyze_concentration(
        self,
        holdings: List[Dict[str, Any]]
    ) -> Optional[InvestmentHypothesis]:
        """Analyze portfolio concentration"""
        
        if not holdings:
            return None
        
        # Calculate total value and weights
        total_value = sum(h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0)) for h in holdings)
        
        if total_value == 0:
            return None
        
        # Find top holding weight
        holding_weights = []
        for h in holdings:
            value = h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0))
            weight = (value / total_value) * 100
            holding_weights.append({
                "symbol": h.get("tradingsymbol", h.get("symbol", "Unknown")),
                "weight": weight,
                "value": value
            })
        
        holding_weights.sort(key=lambda x: x["weight"], reverse=True)
        
        # Check for over-concentration (>30% in single stock)
        if holding_weights and holding_weights[0]["weight"] > 30:
            top = holding_weights[0]
            self._hypothesis_counter += 1
            
            return InvestmentHypothesis(
                hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                title=f"High Concentration Risk in {top['symbol']}",
                description=f"Portfolio is {top['weight']:.1f}% concentrated in {top['symbol']}. "
                           f"This exceeds the recommended 30% single-stock limit and increases idiosyncratic risk.",
                hypothesis_type="risk",
                affected_symbols=[top["symbol"]],
                confidence=0.9,
                supporting_evidence=[
                    f"{top['symbol']} represents {top['weight']:.1f}% of portfolio",
                    "Recommended single-stock allocation is below 30%",
                    "Concentration increases vulnerability to stock-specific events"
                ],
                potential_actions=[
                    f"Consider trimming {top['symbol']} position by {top['weight'] - 25:.1f}%",
                    "Diversify into other sectors or asset classes",
                    "Use protective puts if maintaining position"
                ],
                risk_factors=[
                    "Single stock events can significantly impact portfolio",
                    "Regulatory or company-specific news exposure",
                    "Liquidity risk during market stress"
                ],
                time_horizon="medium-term"
            )
        
        return None
    
    async def _analyze_performance(
        self,
        holdings: List[Dict[str, Any]]
    ) -> List[InvestmentHypothesis]:
        """Analyze holding performance patterns"""
        
        hypotheses = []
        
        # Find big winners and losers
        winners = []
        losers = []
        
        for h in holdings:
            pnl_pct = h.get("pnl_percent", 0)
            symbol = h.get("tradingsymbol", h.get("symbol", "Unknown"))
            
            if pnl_pct > 20:
                winners.append({"symbol": symbol, "pnl": pnl_pct})
            elif pnl_pct < -15:
                losers.append({"symbol": symbol, "pnl": pnl_pct})
        
        # Hypothesis for winners (momentum or profit-taking)
        if winners:
            self._hypothesis_counter += 1
            winners_str = ', '.join([f"{w['symbol']} (+{w['pnl']:.1f}%)" for w in winners[:3]])
            hypotheses.append(InvestmentHypothesis(
                hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                title="Strong Momentum Stocks - Consider Position Management",
                description=f"Several positions showing strong gains: {winners_str}. "
                           f"Consider whether to let winners run or take partial profits.",
                hypothesis_type="bullish",
                affected_symbols=[w["symbol"] for w in winners],
                confidence=0.75,
                supporting_evidence=[
                    f"{len(winners)} stocks with >20% gains",
                    "Strong performance may indicate momentum",
                    "Could also indicate overbought conditions"
                ],
                potential_actions=[
                    "Consider trailing stop-losses to protect gains",
                    "Take partial profits (25-50%) on highest gainers",
                    "Review if fundamentals support continued holding"
                ],
                risk_factors=[
                    "Momentum can reverse quickly",
                    "Winners may be overbought short-term",
                    "Profit-taking pressure near round numbers"
                ],
                time_horizon="short-term"
            ))
        
        # Hypothesis for losers (cut losses or average down)
        if losers:
            self._hypothesis_counter += 1
            losers_str = ', '.join([f"{l['symbol']} ({l['pnl']:.1f}%)" for l in losers[:3]])
            hypotheses.append(InvestmentHypothesis(
                hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                title="Underperforming Positions - Review Required",
                description=f"Several positions showing significant losses: {losers_str}. "
                           f"Review thesis validity and consider position action.",
                hypothesis_type="bearish",
                affected_symbols=[l["symbol"] for l in losers],
                confidence=0.8,
                supporting_evidence=[
                    f"{len(losers)} stocks with >15% losses",
                    "Losses may indicate broken thesis or market headwinds",
                    "Cost of holding losing positions (opportunity cost)"
                ],
                potential_actions=[
                    "Review original investment thesis",
                    "Set stop-loss levels to limit further losses",
                    "Consider tax-loss harvesting if applicable",
                    "Average down only if thesis still valid"
                ],
                risk_factors=[
                    "Losers can continue losing",
                    "Sunk cost fallacy risk",
                    "Opportunity cost of capital"
                ],
                time_horizon="short-term"
            ))
        
        return hypotheses
    
    async def _analyze_sectors(
        self,
        holdings: List[Dict[str, Any]]
    ) -> List[InvestmentHypothesis]:
        """Analyze sector allocation"""
        
        hypotheses = []
        
        # Group by sector (if available)
        sector_values = {}
        total_value = 0
        
        for h in holdings:
            sector = h.get("sector", "Unknown")
            value = h.get("current_value", h.get("quantity", 0) * h.get("last_price", 0))
            sector_values[sector] = sector_values.get(sector, 0) + value
            total_value += value
        
        if total_value == 0:
            return hypotheses
        
        # Check for sector imbalance
        sector_weights = {s: (v / total_value) * 100 for s, v in sector_values.items()}
        
        # Find dominant sector
        dominant_sector = max(sector_weights.items(), key=lambda x: x[1])
        
        if dominant_sector[1] > 40:
            self._hypothesis_counter += 1
            hypotheses.append(InvestmentHypothesis(
                hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                title=f"Sector Concentration in {dominant_sector[0]}",
                description=f"Portfolio is {dominant_sector[1]:.1f}% allocated to {dominant_sector[0]} sector. "
                           f"This creates sector-specific risk exposure.",
                hypothesis_type="neutral",
                affected_symbols=[],  # Would need symbol-to-sector mapping
                confidence=0.85,
                supporting_evidence=[
                    f"{dominant_sector[0]} represents {dominant_sector[1]:.1f}% of portfolio",
                    "Sector concentration increases cyclical risk",
                    "Regulatory changes can impact entire sector"
                ],
                potential_actions=[
                    f"Consider reducing {dominant_sector[0]} exposure",
                    "Add defensive sectors for balance",
                    "Consider sector hedging strategies"
                ],
                risk_factors=[
                    "Sector-wide downturn risk",
                    "Regulatory/policy risk",
                    "Economic cycle sensitivity"
                ],
                time_horizon="medium-term"
            ))
        
        return hypotheses
    
    async def _analyze_risk(
        self,
        holdings: List[Dict[str, Any]],
        market_data: Dict[str, Any] = None
    ) -> List[InvestmentHypothesis]:
        """Analyze portfolio risk factors"""
        
        hypotheses = []
        
        # Calculate aggregate P&L
        total_pnl = sum(h.get("pnl", 0) for h in holdings)
        total_invested = sum(h.get("quantity", 0) * h.get("average_price", 0) for h in holdings)
        
        if total_invested > 0:
            portfolio_return = (total_pnl / total_invested) * 100
            
            # Significant drawdown hypothesis
            if portfolio_return < -10:
                self._hypothesis_counter += 1
                hypotheses.append(InvestmentHypothesis(
                    hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                    title="Portfolio Drawdown Alert",
                    description=f"Portfolio is down {abs(portfolio_return):.1f}% from cost basis. "
                               f"Consider risk management actions.",
                    hypothesis_type="bearish",
                    affected_symbols=[h.get("tradingsymbol", "") for h in holdings if h.get("pnl_percent", 0) < -10],
                    confidence=0.9,
                    supporting_evidence=[
                        f"Portfolio P&L: {portfolio_return:.1f}%",
                        f"Total unrealized loss: ₹{abs(total_pnl):,.0f}",
                        "Market conditions may be unfavorable"
                    ],
                    potential_actions=[
                        "Review and tighten stop-losses",
                        "Reduce position sizes",
                        "Consider hedging with index puts",
                        "Increase cash allocation"
                    ],
                    risk_factors=[
                        "Further market decline risk",
                        "Emotional decision-making",
                        "Margin call risk if using leverage"
                    ],
                    time_horizon="short-term"
                ))
        
        return hypotheses
    
    async def _find_opportunities(
        self,
        holdings: List[Dict[str, Any]],
        market_data: Dict[str, Any] = None
    ) -> List[InvestmentHypothesis]:
        """Find new investment opportunities"""
        
        hypotheses = []
        
        # Check for diversification opportunities
        current_symbols = {h.get("tradingsymbol", h.get("symbol", "")) for h in holdings}
        
        if len(current_symbols) < 5:
            self._hypothesis_counter += 1
            hypotheses.append(InvestmentHypothesis(
                hypothesis_id=f"hyp_{self._hypothesis_counter}_{int(datetime.now().timestamp())}",
                title="Diversification Opportunity",
                description=f"Portfolio has only {len(current_symbols)} holdings. "
                           f"Consider expanding to 10-15 holdings for better diversification.",
                hypothesis_type="opportunity",
                affected_symbols=[],
                confidence=0.8,
                supporting_evidence=[
                    f"Current holdings: {len(current_symbols)}",
                    "Optimal diversification: 10-15 uncorrelated holdings",
                    "Additional holdings reduce idiosyncratic risk"
                ],
                potential_actions=[
                    "Research stocks in uncorrelated sectors",
                    "Consider adding international exposure",
                    "Look at defensive sectors (FMCG, Pharma, Utilities)"
                ],
                risk_factors=[
                    "Over-diversification can reduce returns",
                    "Research time required for new positions",
                    "Transaction costs"
                ],
                time_horizon="medium-term"
            ))
        
        return hypotheses
    
    async def _enhance_with_llm(
        self,
        hypotheses: List[InvestmentHypothesis],
        portfolio_data: Dict[str, Any]
    ) -> List[InvestmentHypothesis]:
        """Use LLM to enhance hypothesis descriptions"""
        
        if not self.llm_client:
            return hypotheses
        
        try:
            # Create summary of hypotheses for LLM
            hyp_summaries = [f"- {h.title}: {h.description[:100]}..." for h in hypotheses[:5]]
            
            prompt = f"""
            Based on these portfolio hypotheses, provide brief, actionable insights:
            
            {chr(10).join(hyp_summaries)}
            
            Portfolio has {len(portfolio_data.get('holdings', []))} holdings.
            
            Provide 2-3 sentences of additional context or priority recommendation.
            Be specific and actionable.
            """
            
            response = await self.llm_client.generate(prompt, max_tokens=200)
            
            # Add LLM insight to first hypothesis
            if response and hypotheses:
                hypotheses[0].supporting_evidence.append(f"AI Insight: {response[:200]}")
        
        except Exception as e:
            logger.error(f"Error enhancing with LLM: {e}")
        
        return hypotheses
    
    def get_active_hypotheses(self) -> List[InvestmentHypothesis]:
        """Get all active hypotheses"""
        return [h for h in self._hypotheses.values() if h.status == "active"]
    
    def invalidate_hypothesis(self, hypothesis_id: str, reason: str = ""):
        """Invalidate a hypothesis"""
        if hypothesis_id in self._hypotheses:
            self._hypotheses[hypothesis_id].status = "invalidated"
            logger.info(f"Hypothesis {hypothesis_id} invalidated: {reason}")
    
    def clear_hypotheses(self):
        """Clear all hypotheses"""
        self._hypotheses.clear()
        self._hypothesis_counter = 0


# Singleton
_hypothesis_generator: Optional[HypothesisGenerator] = None


def get_hypothesis_generator(llm_client=None) -> HypothesisGenerator:
    """Get or create hypothesis generator singleton"""
    global _hypothesis_generator
    if _hypothesis_generator is None:
        _hypothesis_generator = HypothesisGenerator(llm_client)
    return _hypothesis_generator

"""
Portabull - LLaMA API Client
Wrapper for LLaMA model interactions

Supports:
- Local LLaMA model via llama-cpp-python
- Together AI API
- Groq API
- Custom LLaMA API endpoints
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, AsyncIterator
from dataclasses import dataclass
import asyncio
import aiohttp
import json
from loguru import logger


@dataclass
class Message:
    """Chat message"""
    role: str  # system, user, assistant
    content: str


@dataclass
class LLMResponse:
    """Response from LLM"""
    content: str
    model: str
    usage: Dict[str, int]
    finish_reason: str


class BaseLLMClient(ABC):
    """Base class for LLM clients"""
    
    @abstractmethod
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """Generate a response from the LLM"""
        pass
    
    @abstractmethod
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Generate a streaming response from the LLM"""
        pass


class TogetherAIClient(BaseLLMClient):
    """
    Together AI Client for LLaMA models
    
    Together AI provides fast, reliable LLaMA inference
    Good for hackathons and production use
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.together.xyz/v1"
        
        logger.info(f"TogetherAIClient initialized with model: {model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """Generate a response using Together AI"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": False
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Together AI error: {error}")
                    raise Exception(f"Together AI error: {error}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Generate a streaming response"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue


class GroqClient(BaseLLMClient):
    """
    Groq Client for ultra-fast LLaMA inference
    
    Groq provides the fastest LLaMA inference using LPU
    Ideal for real-time chat applications
    """
    
    def __init__(
        self,
        api_key: str,
        model: str = "llama-3.3-70b-versatile"
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = "https://api.groq.com/openai/v1"
        
        logger.info(f"GroqClient initialized with model: {model}")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """Generate a response using Groq"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                if response.status != 200:
                    error = await response.text()
                    logger.error(f"Groq error: {error}")
                    raise Exception(f"Groq error: {error}")
                
                data = await response.json()
                return data["choices"][0]["message"]["content"]
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Generate a streaming response"""
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }
        
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
            "stream": True
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.post(
                f"{self.base_url}/chat/completions",
                headers=headers,
                json=payload
            ) as response:
                async for line in response.content:
                    line = line.decode("utf-8").strip()
                    if line.startswith("data: "):
                        data = line[6:]
                        if data == "[DONE]":
                            break
                        try:
                            chunk = json.loads(data)
                            delta = chunk["choices"][0].get("delta", {})
                            if "content" in delta:
                                yield delta["content"]
                        except json.JSONDecodeError:
                            continue


class LocalLlamaClient(BaseLLMClient):
    """
    Local LLaMA client using llama-cpp-python
    
    Runs LLaMA model locally without API calls
    Requires model file and sufficient GPU/CPU resources
    """
    
    def __init__(
        self,
        model_path: str,
        n_ctx: int = 4096,
        n_gpu_layers: int = -1  # -1 for all layers on GPU
    ):
        self.model_path = model_path
        self.n_ctx = n_ctx
        self.n_gpu_layers = n_gpu_layers
        self._model = None
        
        logger.info(f"LocalLlamaClient initialized with model: {model_path}")
    
    def _get_model(self):
        """Lazy load the model"""
        if self._model is None:
            try:
                from llama_cpp import Llama
                
                self._model = Llama(
                    model_path=self.model_path,
                    n_ctx=self.n_ctx,
                    n_gpu_layers=self.n_gpu_layers,
                    verbose=False
                )
                logger.info("Local LLaMA model loaded")
            except Exception as e:
                logger.error(f"Failed to load local model: {e}")
                raise
        
        return self._model
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """Generate a response using local LLaMA"""
        
        model = self._get_model()
        
        # Format messages into prompt
        prompt = self._format_messages(messages)
        
        # Run in executor to avoid blocking
        loop = asyncio.get_event_loop()
        response = await loop.run_in_executor(
            None,
            lambda: model(
                prompt,
                max_tokens=max_tokens,
                temperature=temperature,
                stop=["</s>", "[/INST]"]
            )
        )
        
        return response["choices"][0]["text"].strip()
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Generate a streaming response"""
        
        model = self._get_model()
        prompt = self._format_messages(messages)
        
        for token in model(
            prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        ):
            yield token["choices"][0]["text"]
    
    def _format_messages(self, messages: List[Dict[str, str]]) -> str:
        """Format messages into LLaMA prompt format"""
        
        prompt_parts = []
        
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            
            if role == "system":
                prompt_parts.append(f"<<SYS>>\n{content}\n<</SYS>>\n\n")
            elif role == "user":
                prompt_parts.append(f"[INST] {content} [/INST]")
            elif role == "assistant":
                prompt_parts.append(f" {content} </s><s>")
        
        return "".join(prompt_parts)


class MockLLMClient(BaseLLMClient):
    """
    Mock LLM client for development and testing
    
    Returns predefined responses for testing without API calls
    """
    
    def __init__(self):
        logger.info("MockLLMClient initialized")
    
    async def generate(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048,
        stream: bool = False
    ) -> str:
        """Generate a mock response"""
        
        # Extract the last user message
        user_message = ""
        for msg in reversed(messages):
            if msg["role"] == "user":
                user_message = msg["content"]
                break
        
        # Generate contextual mock response
        if "portfolio" in user_message.lower():
            return json.dumps({
                "summary": "Your portfolio shows a balanced mix of large-cap stocks with moderate risk.",
                "insights": [
                    "Heavy concentration in IT sector (TCS, Infy)",
                    "Good diversification across banking stocks",
                    "Positive momentum in most holdings"
                ],
                "recommendations": [
                    "Consider adding defensive stocks for balance",
                    "Monitor ICICI Bank for entry opportunities"
                ],
                "risk_factors": [
                    "IT sector exposure to global slowdown",
                    "Banking sector NPA concerns"
                ],
                "confidence": 75
            })
        elif "risk" in user_message.lower():
            return "Based on current market conditions, your portfolio has moderate risk with a beta of 1.1. Key risks include sector concentration and interest rate sensitivity."
        else:
            return f"As your AI portfolio analyst, I've analyzed your question about '{user_message[:50]}...'. Here's my assessment based on current market conditions and your portfolio composition."
    
    async def generate_stream(
        self,
        messages: List[Dict[str, str]],
        temperature: float = 0.7,
        max_tokens: int = 2048
    ) -> AsyncIterator[str]:
        """Generate a mock streaming response"""
        
        response = await self.generate(messages, temperature, max_tokens)
        
        # Simulate streaming by yielding words
        for word in response.split():
            yield word + " "
            await asyncio.sleep(0.05)


def create_llm_client(
    provider: str = "mock",
    api_key: str = None,
    model_path: str = None,
    model: str = None
) -> BaseLLMClient:
    """
    Factory function to create LLM client
    
    Args:
        provider: One of 'together', 'groq', 'local', 'mock'
        api_key: API key for cloud providers
        model_path: Path to local model file
        model: Model name/ID
        
    Returns:
        BaseLLMClient instance
    """
    
    if provider == "together":
        if not api_key:
            raise ValueError("Together AI requires an API key")
        return TogetherAIClient(
            api_key=api_key,
            model=model or "meta-llama/Llama-3.2-11B-Vision-Instruct-Turbo"
        )
    
    elif provider == "groq":
        if not api_key:
            raise ValueError("Groq requires an API key")
        return GroqClient(
            api_key=api_key,
            model=model or "llama-3.3-70b-versatile"
        )
    
    elif provider == "local":
        if not model_path:
            raise ValueError("Local LLaMA requires a model path")
        return LocalLlamaClient(model_path=model_path)
    
    else:
        logger.warning("Using MockLLMClient - for development only")
        return MockLLMClient()

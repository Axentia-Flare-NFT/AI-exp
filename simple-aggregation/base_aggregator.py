from abc import ABC, abstractmethod
import json
import statistics
from typing import Dict, Any, List, Optional, Tuple
from pathlib import Path
import asyncio
import aiohttp

class BaseAggregator(ABC):
    """Abstract base class for AI model response aggregation systems."""
    
    def __init__(self, config_path: str):
        """
        Initialize the aggregator with configuration.
        
        Args:
            config_path: Path to the configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        
    @abstractmethod
    async def fetch_data(self, *args, **kwargs) -> Dict[str, Any]:
        """Fetch the data needed for processing."""
        pass
    
    @abstractmethod
    def prepare_prompt(self, data: Dict[str, Any]) -> List[Dict[str, str]]:
        """Prepare the prompt for the AI models."""
        pass
    
    @abstractmethod
    def extract_value_from_response(self, response: str) -> Optional[float]:
        """Extract the main value from a model's response."""
        pass
    
    def calculate_confidence(self, std_dev: float, values: List[float]) -> float:
        """Calculate confidence score based on standard deviation relative to mean."""
        if not values or len(values) < 2:
            return 0.5
        
        mean_value = statistics.mean(values)
        if mean_value == 0:
            return 0.5
        
        cv = std_dev / mean_value
        confidence = 1.0 - min(cv, 1.0)
        return max(0.1, min(0.9, confidence))
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from file."""
        with open(self.config_path) as f:
            return json.load(f)
    
    async def process_responses(self, responses: Dict[str, str]) -> Dict[str, Any]:
        """Process and aggregate model responses."""
        values = {}
        for model_id, response in responses.items():
            value = self.extract_value_from_response(response)
            if value is not None:
                values[model_id] = value
        
        if not values:
            return {
                "value": 0,
                "confidence": 0,
                "std_dev": 0,
                "error": "No valid values extracted from responses"
            }
        
        mean_value = statistics.mean(values.values())
        std_dev = statistics.stdev(values.values()) if len(values) > 1 else 0
        confidence = self.calculate_confidence(std_dev, list(values.values()))
        
        return {
            "value": mean_value,
            "confidence": confidence,
            "std_dev": std_dev,
            "model_values": values
        }
    
    @abstractmethod
    async def run_aggregation(self, *args, **kwargs) -> Dict[str, Any]:
        """Run the complete aggregation process."""
        pass 
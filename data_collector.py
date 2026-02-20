import logging
from typing import Dict, Optional
from datetime import datetime, timedelta
import requests
import pandas as pd

logger = logging.getLogger(__name__)

class DataCollector:
    """Collects market data from various sources for pricing analysis."""
    
    def __init__(self, api_keys: Dict[str, str], data_sources: list):
        self.api_keys = api_keys
        self.data_sources = data_sources
        self._setup_logger()
        
    def _setup_logger(self) -> None:
        """Configures logging for the module."""
        logger.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        handler = logging.StreamHandler()
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        
    def fetch_data(self, start_time: datetime, end_time: datetime) -> pd.DataFrame:
        """Fetches market data from configured sources within the specified time range."""
        dfs = []
        for source in self.data_sources:
            try:
                df = self._fetch_from_source(source, start_time, end_time)
                dfs.append(df)
            except Exception as e:
                logger.error(f"Failed to fetch data from {source}: {e}")
        
        if not dfs:
            raise ValueError("No data sources returned valid data.")
            
        return pd.concat(dfs)
    
    def _fetch_from_source(self, source: str, start_time: datetime, end_time: datetime) -> Optional[pd.DataFrame]:
        """Fetches data from a specific data source."""
        api_key = self.api_keys.get(source)
        if not api_key:
            logger.error(f"No API key configured for {source}.")
            return None
            
        try:
            response = requests.get(
                self._get_api_url(source),
                params={
                    'start': start_time.isoformat(),
                    'end': end_time.isoformat(),
                    'api_key': api_key
                }
            )
            
            if response.status_code != 200:
                logger.error(f"API request failed for {source}: {response.text}")
                return None
                
            data = response.json()
            df = pd.DataFrame(data['data'])
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            return df
            
        except Exception as e:
            logger.error(f"Error fetching data from {source}: {e}")
            return None
    
    def _get_api_url(self, source: str) -> str:
        """Returns the API URL for a given data source."""
        if source == 'alphavantage':
            return "https://www.alphavantage.co/query"
        elif source == 'quandl':
            return "https://www.quandl.com/api/v3/datasets"
        else:
            logger.error(f"Unsupported data source: {source}")
            raise ValueError("Invalid data source")
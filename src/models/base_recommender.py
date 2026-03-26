from abc import ABC, abstractmethod
import pandas as pd

class BaseRecommender(ABC):
    """
    Abstract base class for all recommendation models.
    Enforces a standard interface so the Hybrid Engine can swap models seamlessly.
    """
    
    @abstractmethod
    def recommend(self, movie_title: str, top_n: int = 5) -> pd.DataFrame:
        """
        Every model must implement this method.
        Returns a DataFrame of recommended movies.
        """
        pass
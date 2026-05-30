import pytest
import pandas as pd
import numpy as np
import faiss
from unittest.mock import MagicMock, patch
from src.models.hybrid_engine import HybridEngine

@pytest.fixture
def mock_df():
    return pd.DataFrame({
        'title': ['Movie A', 'Movie B', 'Movie C'],
        'vote_average': [8.0, 7.0, 6.0],
        'release_date': ['2020-01-01', '2010-01-01', '2000-01-01']
    })

def test_hybrid_engine_init(tmp_path, mock_df):
    df_path = tmp_path / "movies.pkl"
    faiss_path = tmp_path / "movies.faiss"

    mock_df.to_pickle(df_path)

    # Create a dummy faiss index file
    index = faiss.IndexFlatIP(384)
    faiss.write_index(index, str(faiss_path))

    with patch('src.models.hybrid_engine.SentenceTransformer') as mock_st:
        engine = HybridEngine(str(df_path), str(faiss_path))

        expected_df = mock_df.copy()
        expected_df['release_date'] = pd.to_datetime(expected_df['release_date'])

        pd.testing.assert_frame_equal(engine.df, expected_df)
        mock_st.assert_called_once_with('all-MiniLM-L6-v2')

def test_hybrid_engine_get_recommendations(tmp_path, mock_df):
    df_path = tmp_path / "movies.pkl"
    faiss_path = tmp_path / "movies.faiss"
    mock_df.to_pickle(df_path)
    index = faiss.IndexFlatIP(384)
    faiss.write_index(index, str(faiss_path))

    with patch('src.models.hybrid_engine.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype('float32')
        mock_st.return_value = mock_model

        engine = HybridEngine(str(df_path), str(faiss_path))

        # Mock the FAISS index search
        engine.index = MagicMock()
        engine.index.search.return_value = (np.array([[0.9, 0.8]]), np.array([[0, 1]]))

        recs = engine.get_recommendations("test query", top_n=2)

        assert len(recs) == 2
        assert recs == ['Movie A', 'Movie B']

def test_hybrid_engine_filtering(tmp_path, mock_df):
    df_path = tmp_path / "movies.pkl"
    faiss_path = tmp_path / "movies.faiss"
    mock_df.to_pickle(df_path)
    index = faiss.IndexFlatIP(384)
    faiss.write_index(index, str(faiss_path))

    with patch('src.models.hybrid_engine.SentenceTransformer') as mock_st:
        mock_model = MagicMock()
        mock_model.encode.return_value = np.random.rand(1, 384).astype('float32')
        mock_st.return_value = mock_model

        engine = HybridEngine(str(df_path), str(faiss_path))
        engine.index = MagicMock()
        # Return all 3 movies
        engine.index.search.return_value = (np.array([[0.9, 0.8, 0.7]]), np.array([[0, 1, 2]]))

        # Test rating filter
        recs = engine.get_recommendations("query", min_rating=7.5)
        assert recs == ['Movie A']

        # Test year filter
        recs = engine.get_recommendations("query", year_range=(2005, 2015))
        assert recs == ['Movie B']

def test_hybrid_engine_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError):
        HybridEngine(str(tmp_path / "nonexistent.pkl"), str(tmp_path / "nonexistent.faiss"))

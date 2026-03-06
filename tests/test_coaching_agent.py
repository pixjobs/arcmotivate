import pytest
from unittest.mock import patch, MagicMock
from lib.coaching_agent import generate_socratic_stream

@pytest.fixture
def mock_genai_client():
    with patch("lib.coaching_agent.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Setup stream mock
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.parts[0].text = "This is a test question."
        mock_chunk.parts[0].inline_data = None
        
        mock_client.models.generate_content_stream.return_value = [mock_chunk]
        yield mock_client

class TestCoachingAgent:
    def test_generate_socratic_stream_success(self, mock_genai_client):
        superpowers = {"primary": "Explorer"}
        history = [{"role": "user", "text": "I like space."}]
        
        generator = generate_socratic_stream(superpowers, history)
        results = list(generator)
        
        assert len(results) == 1
        assert results[0]["type"] == "text"
        assert results[0]["data"] == "This is a test question."
        mock_genai_client.models.generate_content_stream.assert_called_once()

    def test_generate_socratic_stream_failure(self, mock_genai_client):
        mock_genai_client.models.generate_content_stream.side_effect = Exception("API Error")
        
        generator = generate_socratic_stream({"primary": "Explorer"}, [])
        results = list(generator)
        
        assert len(results) == 1
        assert results[0]["type"] == "text"
        assert "Simulation paused" in results[0]["data"]
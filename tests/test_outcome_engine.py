import pytest
import json
from unittest.mock import patch, MagicMock
from lib.outcome_engine import synthesize_single_tile

@pytest.fixture
def mock_genai_client():
    with patch("lib.outcome_engine.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_tile = {
            "category": "Skill Unlocked",
            "title": "Python Developer",
            "content": "You learned Python.",
            "metadata": ["Code: Python"],
            "image_prompt": "A python programming"
        }
        mock_response.text = json.dumps(mock_tile)
        mock_client.models.generate_content.return_value = mock_response
        yield mock_client

class TestOutcomeEngine:
    def test_synthesize_single_tile(self, mock_genai_client):
        history = [{"role": "user", "text": "I coded in python."}]
        superpowers = {"primary": "Coder"}
        
        result = synthesize_single_tile(history, superpowers)
        
        assert isinstance(result, dict)
        assert result["category"] == "Skill Unlocked"
        mock_genai_client.models.generate_content.assert_called_once()
    
    def test_synthesize_single_tile_failure(self, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = Exception("API format error")
        result = synthesize_single_tile([], {})
        assert result is None
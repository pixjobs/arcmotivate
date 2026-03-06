import pytest
import json
from unittest.mock import patch, MagicMock
from lib.psychology_codex import map_interests_to_superpowers

@pytest.fixture
def mock_genai_client():
    with patch("lib.psychology_codex.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        mock_response = MagicMock()
        mock_response.text = json.dumps({
            "primary": "Creative Builder",
            "secondary": "Hands-On",
            "description": "You love creating things from scratch."
        })
        mock_client.models.generate_content.return_value = mock_response
        
        yield mock_client

class TestPsychologyCodex:
    def test_map_interests_to_superpowers_success(self, mock_genai_client):
        interests = ["drawing", "coding"]
        
        result = map_interests_to_superpowers(interests)
        
        assert isinstance(result, dict)
        assert result["primary"] == "Creative Builder"
        assert result["secondary"] == "Hands-On"
        mock_genai_client.models.generate_content.assert_called_once()

    def test_map_interests_to_superpowers_empty(self, mock_genai_client):
        # Empty interests list returns default without calling API
        result = map_interests_to_superpowers([])
        
        assert result["primary"] == "Explorer"
        mock_genai_client.models.generate_content.assert_not_called()

    def test_map_interests_to_superpowers_failure(self, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = Exception("API error")
        
        result = map_interests_to_superpowers(["gaming"])
        
        # Should gracefully fail to default
        assert result["primary"] == "Explorer"
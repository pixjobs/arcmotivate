import pytest
import json
from unittest.mock import patch, MagicMock
from lib.psychology_codex import map_narrative_to_superpowers

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
    def test_map_narrative_to_superpowers_success(self, mock_genai_client):
        narrative = "I love drawing and building things. I hate sitting still. My best school memory was making a poster for the science fair."

        result = map_narrative_to_superpowers(narrative)

        assert isinstance(result, dict)
        assert result["primary"] == "Creative Builder"
        assert result["secondary"] == "Hands-On"
        mock_genai_client.models.generate_content.assert_called_once()

    def test_map_narrative_to_superpowers_empty(self, mock_genai_client):
        # Empty narrative returns default without calling API
        result = map_narrative_to_superpowers("")

        assert result["primary"] == "Explorer"
        mock_genai_client.models.generate_content.assert_not_called()

    def test_map_narrative_to_superpowers_whitespace(self, mock_genai_client):
        # Whitespace-only narrative also returns default without calling API
        result = map_narrative_to_superpowers("   ")

        assert result["primary"] == "Explorer"
        mock_genai_client.models.generate_content.assert_not_called()

    def test_map_narrative_to_superpowers_failure(self, mock_genai_client):
        mock_genai_client.models.generate_content.side_effect = Exception("API error")

        result = map_narrative_to_superpowers("I like music.")

        # Should gracefully fall back to default
        assert result["primary"] == "Explorer"
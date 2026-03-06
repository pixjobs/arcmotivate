import pytest
import json
from unittest.mock import MagicMock, patch
from lib.storybook_generator import generate_heros_journey_text, generate_pixel_art_illustration, generate_comic_book

@pytest.fixture
def mock_genai_client():
    with patch("lib.storybook_generator.get_client") as mock_get_client:
        mock_client = MagicMock()
        mock_get_client.return_value = mock_client
        
        # Mock for text gen
        mock_response = MagicMock()
        mock_response.text = "A heroic story."
        mock_client.models.generate_content.return_value = mock_response
        
        # Mock for image stream gen
        mock_chunk = MagicMock()
        mock_chunk.parts = [MagicMock()]
        mock_chunk.parts[0].inline_data = MagicMock()
        mock_chunk.parts[0].inline_data.data = b"fake_image_bytes"
        mock_client.models.generate_content_stream.return_value = [mock_chunk]
        
        yield mock_client

class TestStorybookGenerator:
    def test_generate_heros_journey_text(self, mock_genai_client):
        result = generate_heros_journey_text({"superpowers": {"primary": "Explorer"}})
        assert result == "A heroic story."
        mock_genai_client.models.generate_content.assert_called_once()

    def test_generate_pixel_art_illustration(self, mock_genai_client):
        result = generate_pixel_art_illustration("A space scene")
        assert "ZmFrZV9pbWFnZV9ieXRlcw==" in result
        mock_genai_client.models.generate_content_stream.assert_called_once()
        
    def test_generate_comic_book(self, mock_genai_client):
        # Override the text return for json
        mock_genai_client.models.generate_content.return_value.text = json.dumps([
            {"caption": "Panel 1", "image_prompt": "Space"}
        ])
        
        history = [{"role": "user", "text": "start"}]
        result = generate_comic_book(history)
        
        assert len(result) == 1
        assert result[0]["caption"] == "Panel 1"
        assert result[0]["image_b64"] != ""
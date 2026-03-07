from PIL import Image
import io
import base64


def compress_generated_image(image_bytes: bytes, size: int = 320) -> str:
    """
    Compress generated image to small square WebP.
    Returns base64 string.
    """
    try:
        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        # square resize
        img.thumbnail((size, size))

        buffer = io.BytesIO()
        img.save(
            buffer,
            format="WEBP",
            quality=78,
            method=6,
        )

        return base64.b64encode(buffer.getvalue()).decode()

    except Exception:
        return base64.b64encode(image_bytes).decode()
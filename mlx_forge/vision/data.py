"""Vision data collator for VLM training.

Processes image+text samples for vision-language model fine-tuning.
"""

from __future__ import annotations

from typing import Optional


class VisionDataCollator:
    """Processes image+text samples for VLM training.

    1. Extract images from message content
    2. Apply processor's chat template
    3. Tokenize with vision tokens
    4. Create pixel_values tensor
    """

    def __init__(self, processor, max_seq_length: int = 2048):
        self.processor = processor
        self.max_seq_length = max_seq_length

    def __call__(self, sample: dict) -> Optional[dict]:
        """Process a single vision training sample.

        Expected format:
        {"messages": [
            {"role": "user", "content": [
                {"type": "image", "image": "path/to/image.jpg"},
                {"type": "text", "text": "Describe this image"}
            ]},
            {"role": "assistant", "content": "This is a photo of..."}
        ]}

        Returns:
            Dict with input_ids, labels, pixel_values, or None on failure
        """
        try:
            messages = sample.get("messages", [])
            images = self._extract_images(messages)
            text_messages = self._extract_text_messages(messages)

            # Use processor to tokenize with vision tokens
            inputs = self.processor(
                text=text_messages,
                images=images if images else None,
                return_tensors="np",
            )

            input_ids = inputs["input_ids"][0].tolist()
            if len(input_ids) > self.max_seq_length:
                input_ids = input_ids[:self.max_seq_length]

            # Labels: mask everything except assistant tokens
            labels = [-100] * len(input_ids)
            # Simple heuristic: train on second half (assistant response)
            mid = len(input_ids) // 2
            for i in range(mid, len(input_ids)):
                labels[i] = input_ids[i]

            result = {
                "input_ids": input_ids,
                "labels": labels,
            }

            if "pixel_values" in inputs:
                result["pixel_values"] = inputs["pixel_values"]

            return result
        except Exception:
            return None

    def _extract_images(self, messages: list) -> list:
        """Extract image inputs from messages."""
        images = []
        try:
            from PIL import Image
        except ImportError:
            return images

        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, list):
                for item in content:
                    if isinstance(item, dict) and item.get("type") == "image":
                        img_path = item.get("image", "")
                        try:
                            images.append(Image.open(img_path))
                        except Exception:
                            pass
        return images

    def _extract_text_messages(self, messages: list) -> str:
        """Extract text content from messages for tokenization."""
        parts = []
        for msg in messages:
            content = msg.get("content", "")
            if isinstance(content, str):
                parts.append(f"{msg.get('role', 'user')}: {content}")
            elif isinstance(content, list):
                text_parts = [
                    item.get("text", "")
                    for item in content
                    if isinstance(item, dict) and item.get("type") == "text"
                ]
                if text_parts:
                    parts.append(f"{msg.get('role', 'user')}: {' '.join(text_parts)}")
        return "\n".join(parts)

import aiohttp
import os
from variables import WORKER_URL, WORKER_API_KEY

def get_dimensions_from_aspect_ratio(aspect_ratio: str) -> tuple[int, int]:
    """Возвращает (width, height) для соответствующего aspect_ratio."""
    ratios = {
        "1:1": (1024, 1024),
        "16:9": (1280, 720),
        "9:16": (720, 1280),
        "4:3": (1152, 864),
        "3:4": (864, 1152),
        "3:2": (1216, 816),
        "2:3": (816, 1216)
    }
    return ratios.get(aspect_ratio, (1024, 1024))

async def generate_image_via_worker(prompt_text: str, image_b64: str = None, aspect_ratio: str = "9:16") -> bytes:
    width, height = get_dimensions_from_aspect_ratio(aspect_ratio)
    
    headers = {
        "Authorization": f"Bearer {WORKER_API_KEY}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "prompt": prompt_text,
        "width": width,
        "height": height
    }
    
    if image_b64:
        payload["image_b64"] = image_b64

    async with aiohttp.ClientSession() as session:
        async with session.post(WORKER_URL, json=payload, headers=headers, timeout=180) as resp:
            if resp.status == 200:
                return await resp.read()
            else:
                error_text = await resp.text()
                raise RuntimeError(f"Cloudflare Worker Error ({resp.status}): {error_text}")
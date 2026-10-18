"""Render banner.html to docs/images/banner.webp (1600x560).

    uv run --with playwright --with pillow --no-project python docs/banner/render.py [--chromium PATH]

The panels are crops of the running app (maroon-volt, dark); retake them when the UI changes.
"""

import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
OUT = HERE.parent / "images" / "banner.webp"


async def main(chromium: str | None) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium)
        page = await browser.new_page(viewport={"width": 1600, "height": 560}, device_scale_factor=2)
        await page.goto((HERE / "banner.html").as_uri())
        await page.wait_for_load_state("networkidle")
        png = await page.locator(".banner").screenshot(omit_background=True)
        await browser.close()
    # rendered at 2x for clean text, saved at 1x like the other READMEs' banners
    img = Image.open(io.BytesIO(png)).resize((1600, 560), Image.LANCZOS)
    img.save(OUT, "WEBP", quality=90, method=6)
    print(OUT, f"{OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", help="a Chromium binary, if Playwright's own isn't installed")
    asyncio.run(main(ap.parse_args().chromium))

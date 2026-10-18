"""Render banner.html to docs/images/banner.webp (1600x800) and the Registry icon to
docs/images/icon.png (400x400: the banner's lotus on its background).

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
ICON = HERE.parent / "images" / "icon.png"


async def main(chromium: str | None) -> None:
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium)
        page = await browser.new_page(viewport={"width": 1600, "height": 800}, device_scale_factor=2)
        await page.goto((HERE / "banner.html").as_uri())
        await page.wait_for_load_state("networkidle")
        png = await page.locator(".banner").screenshot(omit_background=True)
        # the icon: the banner's own lotus, larger, on the banner's colours
        lotus = (HERE / "banner.html").read_text().split('<svg viewBox="-12 4 88 62"', 1)[1].split("</svg>", 1)[0]
        await page.set_viewport_size({"width": 400, "height": 400})
        await page.set_content(
            '<body style="margin:0"><div id="i" style="width:400px;height:400px;display:grid;place-items:center;'
            'background:radial-gradient(260px 260px at 70% 25%,#3a2747,transparent 70%),linear-gradient(135deg,#1a1320,#2a1d33)">'
            f'<svg viewBox="-12 4 88 62" width="300" height="211"{lotus}</svg></div></body>'
        )
        icon = await page.locator("#i").screenshot()
        await browser.close()
        Image.open(io.BytesIO(icon)).resize((400, 400), Image.LANCZOS).save(ICON, "PNG", optimize=True)
        print(ICON)
    # rendered at 2x for clean text, saved at 1x like the other READMEs' banners
    img = Image.open(io.BytesIO(png)).resize((1600, 800), Image.LANCZOS)
    img.save(OUT, "WEBP", quality=90, method=6)
    print(OUT, f"{OUT.stat().st_size // 1024} KB")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--chromium", help="a Chromium binary, if Playwright's own isn't installed")
    asyncio.run(main(ap.parse_args().chromium))

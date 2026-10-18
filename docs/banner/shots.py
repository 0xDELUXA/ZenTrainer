"""Retake the README screenshots and the banner's panels from a running ZenTrainer.

    uv run --with playwright --with pillow --no-project python docs/banner/shots.py \\
        --url http://127.0.0.1:8188 --run <run_id> --dataset <folder>

Point it at an install holding only demo content (the README's is a knitted-fox plush, made with
Krea 2 Turbo and trained as a Krea 2 character), in the maroon-volt theme, dark. Then render the
banner: python docs/banner/render.py
"""

import argparse
import asyncio
import io
from pathlib import Path

from PIL import Image
from playwright.async_api import async_playwright

HERE = Path(__file__).resolve().parent
IMAGES = HERE.parent / "images"
W, H = 1600, 1000


def save(png: bytes, name: str, width: int = W) -> None:
    img = Image.open(io.BytesIO(png))
    img = img.resize((width, round(img.height * width / img.width)), Image.LANCZOS)
    out = IMAGES / f"{name}.webp"
    img.save(out, "WEBP", quality=88, method=6)
    print(out, img.size, f"{out.stat().st_size // 1024} KB")


async def main(url: str, run: str, dataset: str, chromium: str | None, mask: str | None, chart_run: str | None) -> None:
    app = f"{url}/zentrainer"
    async with async_playwright() as p:
        browser = await p.chromium.launch(executable_path=chromium)
        page = await browser.new_page(viewport={"width": W, "height": H}, device_scale_factor=2, color_scheme="dark")

        async def go(path: str, wait: int = 2500) -> None:
            await page.goto(app + path)
            await page.wait_for_load_state("networkidle")
            await page.wait_for_timeout(wait)
            if mask:  # the demo install's own path, as a reader's would read
                await page.evaluate(
                    """m => { const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
                    for (let n; (n = w.nextNode()); ) if (n.nodeValue.includes(m)) n.nodeValue = n.nodeValue.split(m).join('~/ComfyUI') }""",
                    mask,
                )

        # a run, and the banner's panels from it
        await go(f"/runs/{run}")
        save(await page.screenshot(), "run")
        png = await page.locator(".layout .main section.card >> nth=1").screenshot()
        Image.open(io.BytesIO(png)).save(HERE / "el-samples.png")
        if chart_run:  # the chart can come from a longer run (a character's, with its likeness line)
            await go(f"/runs/{chart_run}")
        png = await page.locator(".layout .main section.card >> nth=0").screenshot()
        Image.open(io.BytesIO(png)).save(HERE / "el-loss.png")

        # New run, filled from the same run (Duplicate)
        await go(f"/runs/new?from={run}")
        save(await page.screenshot(), "new-run")

        # the dataset, then its viewer
        await go(f"/datasets/{dataset}")
        save(await page.screenshot(), "dataset")
        # the page without the app's nav: toolbar, the grid, and the Health pentagon beside it
        nav = await page.locator("aside, nav").first.bounding_box()
        x = (nav["x"] + nav["width"]) if nav else 250
        png = await page.screenshot(clip={"x": x + 8, "y": 20, "width": W - x - 16, "height": 540})
        Image.open(io.BytesIO(png)).save(HERE / "el-dataset.png")
        await page.locator(".card-item img").first.click()
        await page.wait_for_timeout(1500)
        save(await page.screenshot(), "dataset-viewer")
        await page.keyboard.press("Escape")

        # Simple mode: the recipes, and a run as it reads there
        await go("/simple")
        save(await page.screenshot(), "simple")
        await browser.close()


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8188")
    ap.add_argument("--run", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--chromium", help="a Chromium binary, if Playwright's own isn't installed")
    ap.add_argument("--mask", help="the demo install's ComfyUI folder: shown as ~/ComfyUI")
    ap.add_argument("--chart-run", help="take the banner's loss chart from this run instead (only the chart: no samples)")
    a = ap.parse_args()
    asyncio.run(main(a.url, a.run, a.dataset, a.chromium, a.mask, a.chart_run))

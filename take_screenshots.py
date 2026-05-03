#!/usr/bin/env python3
"""Take clean, data-focused dashboard screenshots — no titles, no sidebar.

Strategy: expand Streamlit's scroll container so full_page captures everything,
then measure positions by walking the DOM using cumulative offsetTop, and crop.
"""
import asyncio
from pathlib import Path
from playwright.async_api import async_playwright
from PIL import Image

SCREENSHOTS_DIR = Path(__file__).parent / "screenshots"
BASE_URL = "http://localhost:8501"

HIDE_CSS = """
    [data-testid="stSidebar"] { display: none !important; }
    [data-testid="collapsedControl"] { display: none !important; }
    header[data-testid="stHeader"] { display: none !important; }
    .stDeployButton { display: none !important; }
    #MainMenu { display: none !important; }
    footer { display: none !important; }
    [data-testid="stToolbar"] { display: none !important; }
    .block-container { padding-top: 1rem !important; max-width: 100% !important; }
"""

EXPAND_CSS = """
    [data-testid="stAppViewContainer"],
    [data-testid="stAppViewContainer"] > section,
    [data-testid="stAppViewContainer"] > section > div,
    section.main, section.main > div,
    .stApp,
    [data-testid="stVerticalBlock"],
    [data-testid="stAppViewBlockContainer"],
    section[data-testid="stMain"],
    section[data-testid="stMain"] > div {
        overflow: visible !important;
        max-height: none !important;
        height: auto !important;
    }
"""


async def expand_and_hide(page, hide_h1=True):
    """Apply CSS hiding + expand all scroll containers."""
    await page.add_style_tag(content=HIDE_CSS)
    await page.add_style_tag(content=EXPAND_CSS)
    js = """() => {
        let el = document.querySelector('[data-testid="stAppViewBlockContainer"]')
            || document.querySelector('.block-container');
        while (el && el !== document.body) {
            el.style.overflow = 'visible';
            el.style.maxHeight = 'none';
            el.style.height = 'auto';
            el = el.parentElement;
        }
    }"""
    await page.evaluate(js)
    if hide_h1:
        await page.evaluate("""
            document.querySelectorAll('h1').forEach(el => el.style.display = 'none');
            document.querySelectorAll('h4').forEach(el => {
                if(el.textContent.includes('controlled experiment')) el.style.display = 'none';
            });
        """)
    await page.wait_for_timeout(500)


async def take_screenshots():
    SCREENSHOTS_DIR.mkdir(exist_ok=True)

    async with async_playwright() as p:
        browser = await p.chromium.launch()

        # --- Dashboard page ---
        page = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await page.wait_for_timeout(8000)
        await expand_and_hide(page)
        await page.wait_for_timeout(1000)

        # Full page screenshot
        full_path = SCREENSHOTS_DIR / "dashboard_full.png"
        await page.screenshot(path=str(full_path), full_page=True)
        full_img = Image.open(str(full_path))
        img_w, img_h = full_img.size
        print(f"  Full page: {img_w}x{img_h}")

        # Use Playwright locators to find each section heading's position
        # after overflow expansion, getBoundingClientRect should work
        # because everything is now in a single non-scrolling flow.
        targets = [
            ("hero", "The Headline"),
            ("accuracy", "Accuracy by Incentive Strategy"),
            ("length", "Response Length"),
            ("tone", "How Incentives Change Tone"),
            ("models", "Which Model Is Most Susceptible"),
            ("sentiment", "Emotional Compensation"),
            ("cost", "The Cost Equation"),
            ("bottom_line", "The Bottom Line"),
        ]

        positions = {}
        for key, text in targets:
            # Use Playwright's text locator
            loc = page.locator(f"text={text}").first
            try:
                bbox = await loc.bounding_box(timeout=2000)
                if bbox:
                    positions[key] = bbox["y"]
                    print(f"    {key}: y={bbox['y']:.0f}")
                else:
                    print(f"    {key}: no bbox")
            except Exception as e:
                print(f"    {key}: not found ({e})")

        if not positions:
            print("  No positions found! Trying alternative approach...")
            # Debug: dump some element info
            debug = await page.evaluate("""() => {
                const els = document.querySelectorAll('div[data-testid]');
                return Array.from(els).slice(0, 20).map(el => ({
                    testid: el.getAttribute('data-testid'),
                    offsetTop: el.offsetTop,
                    scrollHeight: el.scrollHeight,
                    clientHeight: el.clientHeight,
                }));
            }""")
            for d in debug:
                print(f"      {d}")

        # Device pixel ratio affects coordinates
        dpr = await page.evaluate("window.devicePixelRatio")
        print(f"  DPR: {dpr}")

        def crop_section(name, start_key, end_key, pad_top=30, pad_bottom=20):
            if start_key not in positions:
                print(f"  SKIP: {start_key} not found")
                return
            sy = int(positions[start_key] * dpr) - int(pad_top * dpr)
            if end_key and end_key in positions:
                ey = int(positions[end_key] * dpr) - int(10 * dpr)
            else:
                ey = img_h
            sy = max(0, sy)
            ey = min(img_h, ey + int(pad_bottom * dpr))
            if ey <= sy + 100:
                print(f"  SKIP: {name} too small ({sy}-{ey})")
                return
            cropped = full_img.crop((0, sy, img_w, ey))
            out_path = SCREENSHOTS_DIR / f"{name}.png"
            cropped.save(str(out_path))
            print(f"  Saved: {name}.png ({cropped.size[0]}x{cropped.size[1]})")

        crop_section("dashboard_hero", "hero", "accuracy", pad_top=120)
        crop_section("dashboard_accuracy", "accuracy", "length")
        crop_section("dashboard_length", "length", "tone")
        crop_section("dashboard_tone", "tone", "models")
        crop_section("dashboard_models", "models", "sentiment")
        crop_section("dashboard_bottom_line", "bottom_line", None, pad_bottom=100)

        await page.close()

        # --- The Experiment page ---
        page2 = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page2.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await page2.wait_for_timeout(4000)
        await page2.evaluate("""
            const labels = document.querySelectorAll('[data-testid="stSidebar"] label');
            for (const l of labels) {
                if (l.textContent.includes('The Experiment')) { l.querySelector('input')?.click(); break; }
            }
        """)
        await page2.wait_for_timeout(3000)
        await expand_and_hide(page2)
        await page2.wait_for_timeout(500)
        await page2.screenshot(path=str(SCREENSHOTS_DIR / "experiment_design.png"), full_page=True)
        print("  Saved: experiment_design.png")
        await page2.close()

        # --- Statistical Tests page ---
        page3 = await browser.new_page(viewport={"width": 1400, "height": 900})
        await page3.goto(BASE_URL, wait_until="networkidle", timeout=30000)
        await page3.wait_for_timeout(4000)
        await page3.evaluate("""
            const labels = document.querySelectorAll('[data-testid="stSidebar"] label');
            for (const l of labels) {
                if (l.textContent.includes('Statistical Tests')) { l.querySelector('input')?.click(); break; }
            }
        """)
        await page3.wait_for_timeout(3000)
        await expand_and_hide(page3)
        await page3.wait_for_timeout(500)
        await page3.screenshot(path=str(SCREENSHOTS_DIR / "statistical_tests.png"), full_page=True)
        print("  Saved: statistical_tests.png")
        await page3.close()

        await browser.close()
        print(f"\nDone!")


if __name__ == "__main__":
    asyncio.run(take_screenshots())

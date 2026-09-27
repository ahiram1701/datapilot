"""Graba un GIF de la app real respondiendo una pregunta (para el README).

Conduce la UI con Playwright, captura la pantalla periódicamente y guarda
solo los frames que cambian: las esperas del LLM se comprimen y el GIF
queda corto aunque la corrida real tarde minutos.

Uso:
    pip install -r scripts/requirements-demo.txt && playwright install chromium
    docker compose up -d        # o backend + frontend locales
    python scripts/record_demo.py --url http://localhost:8080
"""
from __future__ import annotations

import argparse
import hashlib
import io
import sys
import time
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

ROOT = Path(__file__).resolve().parents[1]
CAPTURE_EVERY_S = 0.4
MAX_FRAME_MS = 600       # una pantalla sin cambios nunca dura más que esto
TYPING_FRAME_MS = 60
FINAL_HOLD_MS = 4500
MAX_GIF_MB = 10


class Recorder:
    def __init__(self, page):
        self.page = page
        self.frames: list[Image.Image] = []
        self.durations: list[int] = []
        self._last_hash = None

    def capture(self, duration_ms: int = MAX_FRAME_MS) -> None:
        png = self.page.screenshot()
        h = hashlib.md5(png).hexdigest()
        if h == self._last_hash:  # sin cambios: alarga el frame anterior (con tope)
            self.durations[-1] = min(self.durations[-1] + duration_ms, MAX_FRAME_MS)
            return
        self._last_hash = h
        self.frames.append(Image.open(io.BytesIO(png)).convert("RGB"))
        self.durations.append(duration_ms)

    def hold(self, ms: int) -> None:
        self.durations[-1] = ms


def scroll_to_bottom(page) -> None:
    page.evaluate("window.scrollTo(0, document.body.scrollHeight)")


def run(url: str, dataset: str, question: str, timeout_s: int) -> Recorder:
    with sync_playwright() as pw:
        browser = pw.chromium.launch()
        page = browser.new_page(viewport={"width": 1280, "height": 800},
                                color_scheme="light", device_scale_factor=1)
        rec = Recorder(page)

        page.goto(url)
        page.get_by_role("button", name=dataset).wait_for()
        rec.capture()
        rec.hold(1500)

        page.get_by_role("button", name=dataset).click()
        page.locator(".dataset-info").wait_for()
        rec.capture()
        rec.hold(1800)

        box = page.locator(".ask input")
        box.click()
        for i, ch in enumerate(question):
            page.keyboard.type(ch)
            if i % 3 == 2 or i == len(question) - 1:  # un frame cada 3 letras
                rec.capture(TYPING_FRAME_MS * 3)
        rec.hold(700)

        page.locator(".ask button[type=submit]").click()
        start = time.monotonic()
        last_turn = page.locator(".turn").last
        while True:
            time.sleep(CAPTURE_EVERY_S)
            scroll_to_bottom(page)
            rec.capture()
            if last_turn.locator(".answer").count():
                break
            if last_turn.locator(".error").count():
                msg = last_turn.locator(".error").inner_text()
                browser.close()
                sys.exit(f"La app respondió con error; no se genera el GIF.\n{msg}")
            if time.monotonic() - start > timeout_s:
                browser.close()
                sys.exit(f"Sin respuesta tras {timeout_s}s; no se genera el GIF.")

        # Recorrido final por la respuesta completa
        answer = last_turn.locator(".answer")
        answer.scroll_into_view_if_needed()
        page.evaluate("el => el.scrollIntoView({block: 'start'})", answer.element_handle())
        rec.capture()
        rec.hold(2500)
        while page.evaluate("window.innerHeight + window.scrollY < document.body.scrollHeight - 2"):
            page.mouse.wheel(0, 300)
            time.sleep(0.2)
            rec.capture()
            rec.hold(900)
        rec.hold(FINAL_HOLD_MS)

        print(f"Corrida real: {time.monotonic() - start:.0f}s · {len(rec.frames)} frames")
        browser.close()
        return rec


def save_gif(rec: Recorder, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    for width in (960, 800, 680):
        frames = [f.resize((width, round(f.height * width / f.width)), Image.LANCZOS)
                  .quantize(colors=128, method=Image.Quantize.MEDIANCUT) for f in rec.frames]
        buf = io.BytesIO()
        frames[0].save(buf, format="GIF", save_all=True, append_images=frames[1:],
                       duration=rec.durations, loop=0, optimize=True, disposal=1)
        size_mb = buf.tell() / 1e6
        if size_mb <= MAX_GIF_MB:
            out.write_bytes(buf.getvalue())
            print(f"GIF guardado: {out} ({width}px, {size_mb:.1f} MB, "
                  f"{sum(rec.durations) / 1000:.0f}s)")
            return
        print(f"{width}px pesa {size_mb:.1f} MB; probando más pequeño…")
    sys.exit("No se logró un GIF por debajo del tamaño máximo.")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8080")
    ap.add_argument("--dataset", default="clientes_churn.csv")
    ap.add_argument("--question", default="¿Cómo se puede aumentar la retención de clientes?")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "demo.gif")
    ap.add_argument("--timeout", type=int, default=300, help="segundos máximos esperando la respuesta")
    args = ap.parse_args()
    save_gif(run(args.url, args.dataset, args.question, args.timeout), args.out)


if __name__ == "__main__":
    main()

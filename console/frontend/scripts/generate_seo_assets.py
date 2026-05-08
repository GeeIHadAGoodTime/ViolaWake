from __future__ import annotations

from pathlib import Path
from shutil import copyfile

from PIL import Image, ImageDraw, ImageFont


ROOT = Path(__file__).resolve().parents[1]
PUBLIC = ROOT / "public"
OG_DIR = PUBLIC / "og"


def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf" if bold else "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arialbd.ttf" if bold else "C:/Windows/Fonts/arial.ttf",
    ]
    for candidate in candidates:
        if Path(candidate).exists():
            return ImageFont.truetype(candidate, size=size)
    return ImageFont.load_default()


def wrap(draw: ImageDraw.ImageDraw, text: str, face: ImageFont.ImageFont, max_width: int) -> list[str]:
    words = text.split()
    lines: list[str] = []
    current = ""
    for word in words:
        test = f"{current} {word}".strip()
        width = draw.textbbox((0, 0), test, font=face)[2]
        if width <= max_width or not current:
            current = test
        else:
            lines.append(current)
            current = word
    if current:
        lines.append(current)
    return lines


def draw_og(filename: str, kicker: str, title: str, subtitle: str) -> None:
    OG_DIR.mkdir(parents=True, exist_ok=True)
    img = Image.new("RGB", (1200, 630), "#10131a")
    draw = ImageDraw.Draw(img)

    # Structured background, not a blurred stock-like decoration.
    draw.rectangle((0, 0, 1200, 630), fill="#10131a")
    draw.rectangle((0, 0, 1200, 82), fill="#171b25")
    draw.rectangle((0, 548, 1200, 630), fill="#171b25")
    draw.polygon([(860, 0), (1200, 0), (1200, 630), (1010, 630)], fill="#173f45")
    draw.polygon([(950, 80), (1200, 150), (1200, 500), (1040, 430)], fill="#7e6cff")
    draw.line((72, 156, 680, 156), fill="#38d3bd", width=6)
    draw.line((72, 166, 560, 166), fill="#ffb86b", width=3)

    draw.rounded_rectangle((72, 34, 132, 94), radius=8, fill="#7e6cff")
    draw.text((91, 47), "W", font=font(34, True), fill="#ffffff")
    draw.text((152, 48), "ViolaWake", font=font(34, True), fill="#f4f7fb")

    draw.text((72, 126), kicker.upper(), font=font(24, True), fill="#38d3bd")

    y = 196
    for line in wrap(draw, title, font(58, True), 760)[:3]:
        draw.text((72, y), line, font=font(58, True), fill="#f4f7fb")
        y += 68

    y += 18
    for line in wrap(draw, subtitle, font(30), 790)[:3]:
        draw.text((72, y), line, font=font(30), fill="#c8d1df")
        y += 40

    draw.text((72, 570), "Apache 2.0 SDK | ONNX models | Local inference", font=font(26, True), fill="#f4f7fb")
    img.save(OG_DIR / filename, optimize=True)


def make_favicons() -> None:
    PUBLIC.mkdir(parents=True, exist_ok=True)

    def icon(size: int) -> Image.Image:
        img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
        draw = ImageDraw.Draw(img)
        draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=max(2, size // 8), fill="#7e6cff")
        face = font(max(10, int(size * 0.58)), True)
        text = "W"
        bbox = draw.textbbox((0, 0), text, font=face)
        draw.text(
            ((size - (bbox[2] - bbox[0])) / 2, (size - (bbox[3] - bbox[1])) / 2 - size * 0.05),
            text,
            font=face,
            fill="#ffffff",
        )
        return img

    icon(16).save(PUBLIC / "favicon-16.png")
    icon(32).save(PUBLIC / "favicon-32.png")
    icon(180).save(PUBLIC / "apple-touch-icon-180.png")
    icon(32).save(PUBLIC / "favicon.ico", sizes=[(16, 16), (32, 32)])


def main() -> None:
    draw_og(
        "violawake-og.png",
        "custom wake words",
        "Open-source wake word training",
        "Train ONNX wake word models and run detection locally with an Apache 2.0 SDK.",
    )
    copyfile(OG_DIR / "violawake-og.png", PUBLIC / "og-image.png")
    draw_og(
        "violawake-vs-picovoice.png",
        "Picovoice alternative",
        "ViolaWake vs Picovoice Porcupine",
        "Open training code, ONNX model output, and transparent evaluation for custom wake words.",
    )
    draw_og(
        "violawake-vs-openwakeword.png",
        "OpenWakeWord comparison",
        "ViolaWake builds on OpenWakeWord",
        "A hosted training workflow and SDK layer around the open wake word backbone.",
    )
    draw_og(
        "violawake-vs-snowboy.png",
        "Snowboy replacement",
        "Modern wake word training after Snowboy",
        "Move from deprecated hotword tooling to maintained ONNX wake word models.",
    )
    make_favicons()


if __name__ == "__main__":
    main()

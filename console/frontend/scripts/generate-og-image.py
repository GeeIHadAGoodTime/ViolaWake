"""Generate the ViolaWake Open Graph image.

The image is intentionally generated from code so launch social assets can be
rebuilt without hand-editing a binary file.
"""

from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "public" / "og-image.png"

WIDTH = 1200
HEIGHT = 630


def font(path: str, size: int) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(path, size=size)


FONT_BOLD = font(r"C:\Windows\Fonts\segoeuib.ttf", 88)
FONT_MEDIUM = font(r"C:\Windows\Fonts\segoeui.ttf", 36)
FONT_SMALL = font(r"C:\Windows\Fonts\segoeui.ttf", 27)
FONT_MONO = font(r"C:\Windows\Fonts\consola.ttf", 24)
FONT_MONO_BOLD = font(r"C:\Windows\Fonts\consolab.ttf", 25)


def lerp(a: int, b: int, t: float) -> int:
    return int(a + (b - a) * t)


def gradient_background() -> Image.Image:
    top_left = (26, 26, 46)
    top_right = (22, 33, 62)
    bottom_left = (72, 51, 153)
    bottom_right = (108, 92, 231)
    image = Image.new("RGB", (WIDTH, HEIGHT))
    pixels = image.load()
    for y in range(HEIGHT):
        ty = y / (HEIGHT - 1)
        for x in range(WIDTH):
            tx = x / (WIDTH - 1)
            top = tuple(lerp(top_left[i], top_right[i], tx) for i in range(3))
            bottom = tuple(lerp(bottom_left[i], bottom_right[i], tx) for i in range(3))
            pixels[x, y] = tuple(lerp(top[i], bottom[i], ty) for i in range(3))
    return image


def draw_waveform(draw: ImageDraw.ImageDraw) -> None:
    baseline = 455
    left = 82
    width = 520
    points: list[tuple[float, float]] = []
    for i in range(220):
        x = left + width * (i / 219)
        envelope = max(0.0, math.sin(math.pi * i / 219)) ** 0.85
        y = baseline + (
            math.sin(i * 0.23) * 54 + math.sin(i * 0.071 + 1.7) * 30
        ) * envelope
        points.append((x, y))

    for offset, color, stroke in [
        (0, (0, 206, 201, 225), 7),
        (18, (253, 203, 110, 120), 4),
        (-18, (162, 155, 254, 125), 4),
    ]:
        shifted = [(int(round(x)), int(round(y + offset))) for x, y in points]
        draw.line(shifted, fill=color, width=stroke, joint="curve")

    for x in range(left, left + width + 1, 42):
        draw.line([(x, baseline - 94), (x, baseline + 94)], fill=(255, 255, 255, 28), width=1)


def draw_code_card(draw: ImageDraw.ImageDraw) -> None:
    x0, y0, x1, y1 = 665, 156, 1116, 475
    draw.rounded_rectangle(
        (x0, y0, x1, y1),
        radius=22,
        fill=(13, 18, 35, 220),
        outline=(196, 187, 255, 70),
        width=2,
    )
    draw.rounded_rectangle((x0 + 24, y0 + 24, x1 - 24, y0 + 68), radius=12, fill=(255, 255, 255, 16))
    for index, color in enumerate([(225, 112, 85), (253, 203, 110), (0, 206, 201)]):
        draw.ellipse((x0 + 42 + index * 30, y0 + 39, x0 + 56 + index * 30, y0 + 53), fill=color)

    lines = [
        ("from", (162, 155, 254), " violawake_sdk ", (232, 232, 232), "import", (162, 155, 254)),
        ("WakeDetector", (0, 206, 201), "", (232, 232, 232), "", (232, 232, 232)),
        ("", (232, 232, 232), "", (232, 232, 232), "", (232, 232, 232)),
        ("detector = ", (232, 232, 232), 'WakeDetector("viola")', (253, 203, 110), "", (232, 232, 232)),
        ("detector.stream_mic()", (232, 232, 232), "", (116, 185, 255), "", (232, 232, 232)),
    ]
    y = y0 + 104
    for left, left_color, middle, middle_color, right, right_color in lines:
        x = x0 + 42
        if left:
            draw.text((x, y), left, font=FONT_MONO_BOLD, fill=left_color)
            x += draw.textlength(left, font=FONT_MONO_BOLD)
        if middle:
            draw.text((x, y), middle, font=FONT_MONO, fill=middle_color)
            x += draw.textlength(middle, font=FONT_MONO)
        if right:
            draw.text((x, y), right, font=FONT_MONO_BOLD, fill=right_color)
        y += 43


def main() -> None:
    image = gradient_background().convert("RGBA")
    overlay = Image.new("RGBA", (WIDTH, HEIGHT), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay, "RGBA")

    draw.ellipse((-165, 390, 335, 890), fill=(0, 206, 201, 42))
    draw.ellipse((880, -220, 1390, 290), fill=(253, 203, 110, 34))
    draw.rectangle((0, 0, WIDTH, HEIGHT), outline=(255, 255, 255, 22), width=2)

    draw.text((78, 112), "ViolaWake", font=FONT_BOLD, fill=(255, 255, 255, 255))
    draw.text((84, 218), "Custom Wake Words.", font=FONT_MEDIUM, fill=(232, 232, 232, 232))
    draw.text((84, 265), "Open Source. $0 to Start.", font=FONT_MEDIUM, fill=(232, 232, 232, 232))
    draw.rounded_rectangle((84, 334, 320, 390), radius=18, fill=(0, 206, 201, 38), outline=(0, 206, 201, 130))
    draw.text((112, 345), "Train in minutes", font=FONT_SMALL, fill=(227, 255, 253, 245))

    draw_waveform(draw)
    draw_code_card(draw)

    combined = Image.alpha_composite(image, overlay).convert("RGB")
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    combined.save(OUTPUT, "PNG", optimize=True)
    print(OUTPUT)


if __name__ == "__main__":
    main()

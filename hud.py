"""Compact, width-aware feedback for the live camera view."""
import cv2

WHITE = (235, 240, 245)
CYAN = (255, 220, 90)
GREY = (180, 190, 200)
ORANGE = (60, 170, 255)
FONT = cv2.FONT_HERSHEY_SIMPLEX
KEYS = ("Esc release / R reset / H hide / C duplicate / Z freeze / M outline / "
        "Tab next / X clear all / B background / P clean plate / O occlusion / "
        "T twist / D debug / K help / F fullscreen / E retry / Q quit")


def wrap_text(message, width, scale=.45, max_lines=3):
    """Wrap by measured glyph width and visibly truncate instead of clipping."""
    if width <= 0 or max_lines <= 0:
        return []
    def fits(value):
        return cv2.getTextSize(value, FONT, scale, 1)[0][0] <= width

    lines, line = [], ""
    for word in message.split():
        combined = f"{line} {word}".strip()
        if fits(combined):
            line = combined
            continue
        if line:
            lines.append(line)
            line = ""
        while word and not fits(word):
            cut = max(1, len(word) - 1)
            while cut > 1 and not fits(word[:cut]):
                cut -= 1
            lines.append(word[:cut])
            word = word[cut:]
        line = word
    if line:
        lines.append(line)
    if len(lines) > max_lines:
        lines = lines[:max_lines]
        while lines[-1] and not fits(lines[-1] + "..."):
            lines[-1] = lines[-1][:-1]
        lines[-1] += "..."
    return lines


def panel(frame, top, bottom):
    roi = frame[top:bottom]
    roi[:] = (roi.astype(float) * .24 + (12, 16, 22)).clip(0, 255).astype("uint8")


def draw_lines(frame, lines, x, y, colour=WHITE, scale=.45, leading=19):
    for line in lines:
        cv2.putText(frame, line, (x, y), FONT, scale, colour, 1, cv2.LINE_AA)
        y += leading
    return y


def draw_hud(frame, status, summary, message=None, lost=False, show_keys=False, *, stage=1, hint=None):
    h, w = frame.shape[:2]
    status_lines = wrap_text(status, w - 24, max_lines=2)
    message_lines = wrap_text(message, w - 24, .38, 1 if h < 300 else 2) if message else []
    top_end = 56 + 19 * len(status_lines) + 16 * len(message_lines)
    panel(frame, 0, top_end)
    cv2.line(frame, (12, 31), (w - 12, 31), (65, 70, 78), 1)
    draw_lines(frame, ["TELEKINESIS CV"], 12, 22, CYAN, .48)
    compact = wrap_text(summary, max(40, w - 210), .34, 1)[0]
    size = cv2.getTextSize(compact, FONT, .34, 1)[0][0]
    draw_lines(frame, [compact], w - size - 12, 21, GREY, .34)
    for i, label in enumerate(("1 POINT", "2 PINCH", "3 MOVE"), 1):
        x = 12 + (i - 1) * ((w - 24) // 3)
        draw_lines(frame, [label], x, 48, CYAN if i == stage else GREY, .39)
        if i == stage:
            cv2.line(frame, (x, 53), (x + min(85, (w - 30) // 3), 53), CYAN, 2)
    y = draw_lines(frame, status_lines, 12, 71)
    draw_lines(frame, message_lines, 12, y, CYAN, .38, 16)
    footer = wrap_text(KEYS.replace("K help", "K close help") if show_keys else
                       hint or "Mouse also works / K help / Q quit", w - 24, .34, 5)
    if lost:
        footer = wrap_text("Tracking lost. Last position held; Esc restores reality.", w - 24, .38, 2) + footer
    available = max(1, (h - top_end - 24) // 16)
    if len(footer) > available:
        footer = footer[:available]
        footer[-1] = wrap_text(footer[-1] + " / K help", w - 24, .34, 1)[0]
    footer_top = max(top_end + 12, h - 10 - 16 * len(footer))
    panel(frame, footer_top, h)
    draw_lines(frame, footer, 12, footer_top + 13, ORANGE if lost else GREY, .34, 16)
    return top_end

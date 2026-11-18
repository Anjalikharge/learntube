"""Small shared helpers."""
import re


def fmt_time(sec: float) -> str:
    sec = int(sec)
    h, r = divmod(sec, 3600)
    m, s = divmod(r, 60)
    return f"{h}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"


def parse_time(label: str) -> int:
    sec = 0
    for p in label.split(":"):
        sec = sec * 60 + int(p)
    return sec


def ts_url(video_id: str, seconds: float) -> str:
    return f"https://youtu.be/{video_id}?t={int(seconds)}"


_TS = re.compile(r"\[(\d{1,2}:\d{2}(?::\d{2})?)\]")


def linkify(markdown: str, video_id) -> str:
    """Turn [mm:ss] markers in text into clickable YouTube timestamp links."""
    if not video_id:
        return markdown
    return _TS.sub(lambda m: f"[{m.group(1)}]({ts_url(video_id, parse_time(m.group(1)))})", markdown)

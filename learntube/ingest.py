"""Data acquisition: URL -> video id -> timestamped transcript (captions, with Whisper fallback)."""
import json
import re
import tempfile
import urllib.parse
import urllib.request

_ID = re.compile(r"(?:v=|youtu\.be/|shorts/|embed/|live/)([A-Za-z0-9_-]{11})")


def get_video_id(url: str) -> str:
    url = url.strip()
    if re.fullmatch(r"[A-Za-z0-9_-]{11}", url):
        return url
    m = _ID.search(url)
    if not m:
        raise ValueError("Invalid YouTube URL")
    return m.group(1)


def fetch_transcript(url: str, languages=("en", "en-IN", "en-US", "hi", "mr")):
    """Return (video_id, segments). segments = [{text, start, duration}]."""
    from youtube_transcript_api import YouTubeTranscriptApi
    vid = get_video_id(url)
    api = YouTubeTranscriptApi()
    try:
        fetched = api.fetch(vid, languages=list(languages))
    except Exception:
        try:  # fall back to whichever transcript exists
            fetched = next(iter(api.list(vid))).fetch()
        except Exception as e:
            raise RuntimeError("Could not fetch a transcript (captions disabled, video private, "
                               f"or YouTube blocked this network). Details: {e}")
    segs = [{"text": s.text, "start": float(s.start), "duration": float(s.duration)} for s in fetched]
    if not segs:
        raise RuntimeError("Transcript is empty.")
    return vid, segs


def segments_from_text(text: str):
    """Manual fallback: pasted transcript. Lines like '[01:23] text' keep their timestamps."""
    segs, t = [], 0.0
    ts = re.compile(r"^\s*\[?(\d{1,2}):(\d{2})(?::(\d{2}))?\]?\s*(.*)$")
    lines = [l for l in text.splitlines() if l.strip()]
    stamped = sum(1 for l in lines if ts.match(l)) > len(lines) / 2
    if stamped:
        for l in lines:
            m = ts.match(l)
            if m:
                a, b, c, rest = m.groups()
                t = int(a) * 3600 + int(b) * 60 + int(c) if c else int(a) * 60 + int(b)
                segs.append({"text": rest, "start": float(t), "duration": 0.0})
            elif segs:
                segs[-1]["text"] += " " + l.strip()
        return segs
    return [{"text": text, "start": 0.0, "duration": 0.0}]


def fetch_title(video_id: str) -> str:
    """Video title via YouTube oEmbed (no API key). Returns '' on any failure."""
    try:
        u = "https://www.youtube.com/oembed?format=json&url=" + urllib.parse.quote(
            f"https://www.youtube.com/watch?v={video_id}")
        with urllib.request.urlopen(u, timeout=8) as r:
            return json.loads(r.read()).get("title", "")
    except Exception:
        return ""


def transcribe_with_whisper(url: str, model_size: str = "small"):
    """Fallback for videos without captions: download audio (yt-dlp) and transcribe (faster-whisper)."""
    try:
        import yt_dlp
        from faster_whisper import WhisperModel
    except ImportError:
        raise RuntimeError("Whisper fallback needs: pip install yt-dlp faster-whisper")
    vid = get_video_id(url)
    with tempfile.TemporaryDirectory() as td:
        opts = {"format": "bestaudio/best", "outtmpl": f"{td}/audio.%(ext)s", "quiet": True}
        with yt_dlp.YoutubeDL(opts) as y:
            info = y.extract_info(f"https://www.youtube.com/watch?v={vid}", download=True)
            path = y.prepare_filename(info)
        model = WhisperModel(model_size, compute_type="int8")
        segs, _ = model.transcribe(path)
        out = [{"text": s.text.strip(), "start": float(s.start), "duration": float(s.end - s.start)} for s in segs]
    if not out:
        raise RuntimeError("Whisper produced no text.")
    return vid, out

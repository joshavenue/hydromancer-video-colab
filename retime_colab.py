"""Colab / Linux CUDA port of Hydromancer's tools/retime.py.

Re-times a MiniMax-H3 clip so Jojo's mouth follows the reference narration WAV
when H3 stretches or squeezes speech timing inside a clip.

Replaces Mac-only `mlx_whisper` with `faster-whisper` (word timestamps) and
accepts explicit CLI paths rather than hardcoded Mac paths.

Usage:
    python retime_colab.py --clip local/j01.mp4 --ref-audio j01.wav --out local_retimed/j01.mp4
"""
import argparse
import difflib
import re
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np
import soundfile as sf
from faster_whisper import WhisperModel


def extract_16k_wav(src: Path, dst: Path) -> None:
    subprocess.run(
        ["ffmpeg", "-v", "error", "-y", "-i", str(src), "-vn", "-ac", "1", "-ar", "16000", str(dst)],
        check=True,
    )


def transcribe_words(model: WhisperModel, wav_path: Path, language: str = "en") -> list[dict]:
    segments, _ = model.transcribe(
        str(wav_path),
        language=language,
        word_timestamps=True,
        condition_on_previous_text=False,
    )
    words = []
    for seg in segments:
        if not seg.words:
            continue
        for w in seg.words:
            words.append({"w": w.word, "s": float(w.start), "e": float(w.end)})
    return words


def norm(word: str) -> str:
    return re.sub(r"[^a-z0-9]", "", word.lower())


def retime_clip(
    clip_path: Path,
    ref_audio_path: Path,
    out_path: Path,
    model: WhisperModel | None = None,
    model_size: str = "large-v3-turbo",
    device: str = "cuda",
    compute_type: str = "float16",
    mux_ref_audio: bool = True,
) -> dict:
    if model is None:
        model = WhisperModel(model_size, device=device, compute_type=compute_type)

    out_path.parent.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory() as tmp:
        tmp_dir = Path(tmp)
        clip_wav = tmp_dir / "clip.wav"
        ref_wav = tmp_dir / "ref.wav"
        video_only = tmp_dir / "retimed_video.mp4"

        extract_16k_wav(clip_path, clip_wav)
        extract_16k_wav(ref_audio_path, ref_wav)

        narr = transcribe_words(model, ref_wav)
        cw = transcribe_words(model, clip_wav)

        a = [norm(w["w"]) for w in narr]
        b = [norm(w["w"]) for w in cw]

        anch = [(0.0, 0.0)]
        for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks():
            for k in range(m.size):
                n_w, c_w = narr[m.a + k], cw[m.b + k]
                anch += [(n_w["s"], c_w["s"]), (n_w["e"], c_w["e"])]

        anch.sort()
        mono = [anch[0]]
        for p in anch[1:]:
            if p[0] > mono[-1][0] + 1e-3 and p[1] > mono[-1][1] + 1e-3:
                mono.append(p)

        nt = np.array([p[0] for p in mono])
        ct = np.array([p[1] for p in mono])

        cap = cv2.VideoCapture(str(clip_path))
        fps = cap.get(cv2.CAP_PROP_FPS) or 24.0
        w_px = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h_px = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frames = []
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            raise RuntimeError(f"Could not read any frames from {clip_path}")

        clip_dur = len(frames) / fps
        out_dur = float(nt[-1] + max(0.0, clip_dur - ct[-1]))

        ff = subprocess.Popen(
            [
                "ffmpeg", "-v", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "bgr24",
                "-s", f"{w_px}x{h_px}", "-r", str(fps), "-i", "-",
                "-c:v", "libx264", "-crf", "14", "-pix_fmt", "yuv420p",
                str(video_only),
            ],
            stdin=subprocess.PIPE,
        )
        assert ff.stdin is not None
        for i in range(int(round(out_dur * fps))):
            t = i / fps
            c = float(np.interp(t, nt, ct)) if t <= nt[-1] else float(ct[-1] + (t - nt[-1]))
            idx = min(len(frames) - 1, max(0, int(round(c * fps))))
            ff.stdin.write(frames[idx].tobytes())
        ff.stdin.close()
        ff.wait()
        if ff.returncode != 0:
            raise RuntimeError(f"ffmpeg retiming failed for {clip_path}")

        if mux_ref_audio:
            subprocess.run(
                [
                    "ffmpeg", "-v", "error", "-y",
                    "-i", str(video_only), "-i", str(ref_audio_path),
                    "-map", "0:v:0", "-map", "1:a:0",
                    "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                    str(out_path),
                ],
                check=True,
            )
        else:
            video_only.replace(out_path)

        drifts = [round(float(c - n), 3) for n, c in mono]
        max_drift = max(drifts, key=abs) if drifts else 0.0
        return {
            "clip": str(clip_path.name),
            "matched_anchors": len(mono),
            "ref_words": len(narr),
            "clip_words": len(cw),
            "max_drift_sec": max_drift,
            "clip_dur_sec": round(clip_dur, 2),
            "out_dur_sec": round(out_dur, 2),
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Retime an H3 clip using faster-whisper word alignment.")
    parser.add_argument("--clip", required=True, type=Path)
    parser.add_argument("--ref-audio", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--model", default="large-v3-turbo")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--compute-type", default="float16")
    parser.add_argument("--video-only", action="store_true")
    args = parser.parse_args()

    stats = retime_clip(
        args.clip,
        args.ref_audio,
        args.out,
        model_size=args.model,
        device=args.device,
        compute_type=args.compute_type,
        mux_ref_audio=not args.video_only,
    )
    print(stats)


if __name__ == "__main__":
    main()

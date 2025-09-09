# tasks/asr_chirp2.py
import os
import asyncio
import tempfile
import shlex
import subprocess
import uuid
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from google.cloud import speech_v2 as speech
from google.api_core.client_options import ClientOptions
from google.cloud import storage

# ---- Config (env) ------------------------------------------------------------
SPEECH_ENABLE = os.getenv("SPEECH_ENABLE", "1") == "1"
# chirp_2: use REGIONAL location (e.g. us-central1 / europe-west4 / asia-southeast1)
SPEECH_LOCATION = os.getenv("SPEECH_LOCATION", "us-central1")
SPEECH_RECOGNIZER = os.getenv("SPEECH_RECOGNIZER", "_")   # "_" = default recognizer
SPEECH_MODEL = os.getenv("SPEECH_MODEL", "chirp_2")
SPEECH_LANGUAGE_CODES: List[str] = [
    s.strip() for s in os.getenv("SPEECH_LANGUAGE_CODES", "auto").split(",") if s.strip()
]

# Regional endpoint for v2
_SPEECH_ENDPOINT = f"{SPEECH_LOCATION}-speech.googleapis.com"
_speech_client = speech.SpeechAsyncClient(
    client_options=ClientOptions(api_endpoint=_SPEECH_ENDPOINT)
)

# Reusable GCS client
_gcs = storage.Client()


# ---- Paths / Small utilities -------------------------------------------------
def recognizer_path(project_id: str) -> str:
    return f"projects/{project_id}/locations/{SPEECH_LOCATION}/recognizers/{SPEECH_RECOGNIZER}"

def is_audio_attachment(att: Dict) -> bool:
    return str(att.get("mime_type", "")).startswith("audio/")

def to_gcs_uri(att: Dict) -> str:
    return f"gs://{att.get('bucket')}/{att.get('object_path')}"

def guess_mime(att: Dict) -> str:
    """Prefer explicit mime_type; fall back to file extension; default octet-stream."""
    mt = (att.get("mime_type") or "").strip()
    if mt:
        return mt
    # best-effort by extension
    name = att.get("file_name", "")
    ext = Path(name).suffix.lower()
    if ext == ".webm": return "audio/webm"
    if ext in (".m4a", ".mp4", ".aac"): return "audio/mp4"
    if ext == ".mp3": return "audio/mpeg"
    if ext in (".wav", ".wave"): return "audio/wav"
    if ext == ".flac": return "audio/flac"
    if ext == ".ogg": return "audio/ogg"
    return "application/octet-stream"

def _ffmpeg_available() -> bool:
    try:
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# ---- Normalization: any input → mono 16 kHz WAV in GCS -----------------------
async def normalize_audio_to_gcs_wav(
    att: Dict,
    *,
    sample_rate_hz: int = 16000,
    channels: int = 1,
    force: bool = True,
) -> Tuple[str, Dict]:
    """
    Download gs://bucket/object to /tmp, convert to mono <sample_rate_hz> WAV with ffmpeg,
    upload to gs://bucket/normalized/<uuid>_<name>.wav, return (normalized_gs_uri, meta).

    If ffmpeg is unavailable OR conversion fails, returns original gs:// URI with
    meta.normalized=False so caller can still attempt recognition.

    att shape: {bucket, object_path, mime_type, file_name}
    """
    bucket = att.get("bucket")
    obj = att.get("object_path")
    if not bucket or not obj:
        raise ValueError("Attachment must include 'bucket' and 'object_path'.")

    src_blob = _gcs.bucket(bucket).blob(obj)
    await asyncio.to_thread(src_blob.reload)  # fetch size/metadata

    src_uri = f"gs://{bucket}/{obj}"
    mime = guess_mime(att)

    if not _ffmpeg_available():
        return src_uri, {"normalized": False, "reason": "ffmpeg_missing", "mime": mime}

    # Always normalize (safest for AAC/M4A); set force=False if you'd like to pass-through "safe" formats
    with tempfile.TemporaryDirectory() as td:
        in_path = Path(td) / Path(obj).name
        out_path = Path(td) / (Path(obj).stem + ".wav")

        # download to /tmp
        data = await asyncio.to_thread(src_blob.download_as_bytes)
        in_path.write_bytes(data)

        # run ffmpeg
        cmd = f'ffmpeg -y -i "{in_path}" -ac {channels} -ar {sample_rate_hz} -f wav "{out_path}"'
        try:
            subprocess.run(shlex.split(cmd), capture_output=True, check=True)
        except subprocess.CalledProcessError as e:
            return src_uri, {
                "normalized": False,
                "error": (e.stderr.decode("utf-8", "ignore")[:1000] if e.stderr else "ffmpeg_failed"),
                "mime": mime,
            }

        # upload normalized wav under 'normalized/' prefix
        norm_name = f"normalized/{uuid.uuid4()}_{out_path.name}"
        dst_blob = _gcs.bucket(bucket).blob(norm_name)
        await asyncio.to_thread(dst_blob.upload_from_filename, str(out_path), content_type="audio/wav")

        norm_uri = f"gs://{bucket}/{norm_name}"
        return norm_uri, {
            "normalized": True,
            "source_uri": src_uri,
            "dst_object_path": norm_name,
            "mime": mime,
            "sample_rate_hz": sample_rate_hz,
            "channels": channels,
        }


# ---- Main callable: transcribe a gs:// URI with Chirp 2 ----------------------
async def transcribe_gcs_with_chirp2(
    project_id: str,
    gcs_uri: str,
    *,
    language_codes: Optional[List[str]] = None,
    model: Optional[str] = None,
    enable_punctuation: bool = True,
    phrase_hints: Optional[List[str]] = None,
) -> Dict:
    """
    One-shot transcription for short audio using Speech-to-Text v2 (chirp_2).
    Returns: {text, confidence, model, requested_language_codes, detected_language_code, gcs_uri}
    """
    language_codes = language_codes or SPEECH_LANGUAGE_CODES
    model = model or SPEECH_MODEL

    # Chirp 2 supports a single language or ["auto"] only
    if model.startswith("chirp") and language_codes and language_codes != ["auto"] and len(language_codes) > 1:
        raise ValueError(
            "chirp_2 only supports a single language or ['auto']; "
            "use 'latest_short/long' in multi-region (global/us/eu) for multiple language candidates."
        )

    adaptation = None
    if phrase_hints:
        # Inline phrase set (compatible with v2)
        adaptation = speech.SpeechAdaptation(
            phrase_sets=[
                speech.SpeechAdaptation.AdaptationPhraseSet(
                    inline_phrase_set=speech.PhraseSet(
                        phrases=[speech.Phrase(value=p) for p in phrase_hints]
                    )
                )
            ]
        )

    cfg = speech.RecognitionConfig(
        auto_decoding_config=speech.AutoDetectDecodingConfig(),
        model=model,
        language_codes=language_codes,  # e.g. ["auto"] for auto-detect or ["es-US"]
        features=speech.RecognitionFeatures(
            enable_automatic_punctuation=enable_punctuation,
        ),
        adaptation=adaptation,
    )

    req = speech.RecognizeRequest(
        recognizer=recognizer_path(project_id),
        config=cfg,
        uri=gcs_uri,  # service reads directly from GCS
    )

    resp = await _speech_client.recognize(request=req)

    transcripts: List[str] = []
    confidences: List[float] = []
    detected_lang: Optional[str] = None

    for r in resp.results:
        if r.alternatives:
            a0 = r.alternatives[0]
            if a0.transcript:
                transcripts.append(a0.transcript.strip())
            # confidence may be absent for some models
            if getattr(a0, "confidence", None) is not None:
                confidences.append(a0.confidence)
        # Pick the first reported language_code as the detected language
        if not detected_lang and getattr(r, "language_code", None):
            detected_lang = r.language_code

    text = " ".join(transcripts).strip()
    conf = (sum(confidences) / len(confidences)) if confidences else None

    return {
        "text": text,
        "confidence": conf,
        "model": model,
        "requested_language_codes": language_codes,
        "detected_language_code": detected_lang,
        "gcs_uri": gcs_uri,
    }


# ---- Convenience: normalize then transcribe a single attachment --------------
async def transcribe_normalized_attachment(
    project_id: str,
    attachment: Dict,
    *,
    language_codes: Optional[List[str]] = None,
    model: Optional[str] = None,
    enable_punctuation: bool = True,
    phrase_hints: Optional[List[str]] = None,
) -> Dict:
    """
    Helper for workers: normalize the audio attachment to WAV in GCS, then transcribe it.
    Returns the same dict as transcribe_gcs_with_chirp2(), plus {normalized: bool, normalized_uri?, source_uri?}.
    """
    norm_uri, meta = await normalize_audio_to_gcs_wav(attachment)
    out = await transcribe_gcs_with_chirp2(
        project_id,
        norm_uri,
        language_codes=language_codes,
        model=model,
        enable_punctuation=enable_punctuation,
        phrase_hints=phrase_hints,
    )
    out.update({
        "normalized": meta.get("normalized", False),
        "normalized_uri": norm_uri if meta.get("normalized") else None,
        "source_uri": meta.get("source_uri") or to_gcs_uri(attachment),
        "normalization_meta": meta,
    })
    return out

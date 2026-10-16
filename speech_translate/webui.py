"""Browser demo (Gradio).

Two ways in:

* **Translate** -- record or upload a clip and get transcript, translation and
  synthesised speech back. Works on a hosted Space, where the server has no
  microphone of its own, because the browser does the capture.
* **Live** -- streams microphone audio and shows a provisional caption while
  you are still talking, then the final translation (and its speech) as each
  sentence ends. The model work runs on a per-session background thread (see
  :mod:`speech_translate.live`), so capture never waits for translation.

The heavy models are loaded once and shared. Only the TTS voice is swapped when
the target language changes, because ASR and MT are language-independent and
reloading a 1.3B-parameter translator per request would dominate latency.

    python -m speech_translate.webui --share
"""

from __future__ import annotations

import argparse
import logging
import tempfile
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .config import ASRSettings, MTSettings, Settings, TTSSettings
from .languages import AUTO, languages_with_speech, supported_languages
from .live import LiveTranslator

logger = logging.getLogger(__name__)

__all__ = ["build_demo", "main"]

SAMPLE_RATE = 16_000

_DESCRIPTION = """
# Real-Time Speech Translator

Speak into the mic in one language and get it back in another. Everything runs
locally, nothing is sent to an external API.

`faster-whisper` (speech recognition) -> `NLLB-200` (translation, 200 languages) -> `Piper` (speech)
"""


# Lets the Live tab listen to another browser tab (a video, a call) instead of
# the microphone. Gradio's recorder asks for audio through getUserMedia; while
# the "listen to a tab" box is ticked, that request is answered with the audio
# of a tab the user picks through getDisplayMedia, so the rest of the pipeline
# cannot tell the difference. Chrome and Edge support tab audio; Safari and
# Firefox do not share audio this way.
#
# Hearing the original at full volume drowns out the translation, so where the
# browser supports it (suppressLocalAudioPlayback, Chrome 109+) the tab is
# silenced at the source and replayed here through a gain node: its volume is
# a slider, and it is turned down further while the translation is speaking.
_TAB_CAPTURE_JS = """
<script>
(() => {
  const media = navigator.mediaDevices;
  if (!media || !media.getUserMedia || !media.getDisplayMedia) return;
  const getUserMedia = media.getUserMedia.bind(media);
  let tabStream = null;
  let pending = null;
  window.__speechTranslateTab = false;
  // Volume of the original tab, 0..1, and how much of it is left while the
  // translation speaks.
  window.__speechTranslateOriginalVolume = 0.2;
  const DUCK = 0.35;
  const canSuppress = !!(media.getSupportedConstraints &&
    media.getSupportedConstraints().suppressLocalAudioPlayback);
  let replay = null;  // {context, gain}
  let translationPlaying = false;

  const applyVolume = () => {
    if (!replay) return;
    const level = window.__speechTranslateOriginalVolume * (translationPlaying ? DUCK : 1);
    replay.gain.gain.setTargetAtTime(level, replay.context.currentTime, 0.08);
  };
  window.__speechTranslateSetVolume = (percent) => {
    window.__speechTranslateOriginalVolume = Math.max(0, Math.min(1, percent / 100));
    applyVolume();
  };
  // The only other audio on the page is the spoken translation. Gradio's
  // player plays it through a media element that is not in the document, so
  // document-level listeners never hear it; hook play() on every element.
  const playing = new Set();
  const track = (element) => {
    if (element.__speechTranslateTracked) return;
    element.__speechTranslateTracked = true;
    const update = () => {
      if (!element.paused && !element.ended) playing.add(element);
      else playing.delete(element);
      translationPlaying = playing.size > 0;
      applyVolume();
    };
    for (const event of ["playing", "pause", "ended", "emptied"]) {
      element.addEventListener(event, update);
    }
  };
  const play = HTMLMediaElement.prototype.play;
  HTMLMediaElement.prototype.play = function (...args) {
    track(this);
    return play.apply(this, args);
  };

  const live = (stream) =>
    stream && stream.getAudioTracks().some((t) => t.readyState === "live");

  media.getUserMedia = async (constraints) => {
    if (!window.__speechTranslateTab || !constraints || !constraints.audio) {
      return getUserMedia(constraints);
    }
    // The recorder asks more than once per recording, sometimes in parallel;
    // share one picker and one tab between those requests.
    if (live(tabStream)) return tabStream;
    if (!pending) pending = pickTab().finally(() => { pending = null; });
    tabStream = await pending;
    return tabStream;
  };

  async function pickTab() {
    const shared = await media.getDisplayMedia({
      video: true,
      audio: {
        echoCancellation: false,
        noiseSuppression: false,
        autoGainControl: false,
        suppressLocalAudioPlayback: true,
      },
      selfBrowserSurface: "exclude",
      systemAudio: "exclude",
    });
    const audio = shared.getAudioTracks();
    if (!audio.length) {
      shared.getTracks().forEach((t) => t.stop());
      throw new DOMException(
        "No tab audio was shared. Pick a Chrome tab and keep 'Share tab audio' on.",
        "NotAllowedError"
      );
    }
    // Chrome requires video to share a tab; only the audio is used. When
    // the recorder stops the audio, end the share as well.
    const video = shared.getVideoTracks();
    if (canSuppress) {
      // The tab is now silent at the source; play it back here, quieter.
      const context = new AudioContext();
      const gain = context.createGain();
      context.createMediaStreamSource(new MediaStream(audio)).connect(gain);
      gain.connect(context.destination);
      replay = { context, gain };
      gain.gain.value = 0;
      applyVolume();
    }
    audio[0].addEventListener("ended", () => {
      video.forEach((t) => t.stop());
      if (replay) { replay.context.close(); replay = null; }
    });
    return new MediaStream(audio);
  }
})();
</script>
"""


def _theme():
    """Slate/neutral theme.

    Gradio's default palettes lean heavily on violet, which reads as generic
    AI-demo. Slate with system fonts looks closer to a normal developer tool.
    """
    import gradio as gr

    return gr.themes.Base(
        primary_hue=gr.themes.colors.slate,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.gray,
        font=["system-ui", "-apple-system", "Segoe UI", "Roboto", "sans-serif"],
        font_mono=["ui-monospace", "SFMono-Regular", "Consolas", "monospace"],
    ).set(
        button_primary_background_fill="*neutral_800",
        button_primary_background_fill_hover="*neutral_900",
        button_primary_text_color="white",
        block_title_text_weight="600",
    )


def launch_demo(demo, **kwargs):  # noqa: ANN001, ANN201
    """Launch ``demo`` with the app's theme and footer.

    Gradio 6 moved ``theme`` from ``gr.Blocks`` to ``launch()``, so every entry
    point (the CLI and the Spaces ``app.py``) goes through here to look the same.
    """
    kwargs.setdefault("theme", _theme())
    kwargs.setdefault("head", _TAB_CAPTURE_JS)
    # Hide the "Use via API" footer link.
    kwargs.setdefault("footer_links", ["gradio", "settings"])
    return demo.launch(**kwargs)


@dataclass
class _ModelCache:
    """Process-wide model cache, guarded because Gradio serves concurrently."""

    settings: Settings
    lock: threading.Lock = field(default_factory=threading.Lock)
    #: Held around model calls: the clip tab and live sessions share models.
    infer_lock: threading.RLock = field(default_factory=threading.RLock)
    _pipeline: object | None = None
    _tts_language: str = ""

    def pipeline(self, target_language: str, speak: bool):
        from .pipeline import Pipeline
        from .tts import create_tts_backend

        with self.lock:
            if self._pipeline is None:
                self.settings = self.settings.with_languages(tgt=target_language)
                self._pipeline = Pipeline(self.settings)
                self._pipeline.warmup()
                self._tts_language = target_language

            pipeline = self._pipeline
            pipeline.settings.tgt = target_language

            needs_voice = speak and (
                self._tts_language != target_language or pipeline._tts is None
            )
            if needs_voice:
                if pipeline._tts is not None:
                    pipeline._tts.close()
                backend = create_tts_backend(target_language, self.settings.tts)
                try:
                    # Warm the new voice here rather than inside the user's
                    # first request; Piper's first synthesis builds its graph.
                    backend.synthesize("Hola.")
                except Exception:
                    logger.debug("Voice warm-up failed (harmless)", exc_info=True)
                pipeline._tts = backend
                self._tts_language = target_language
            return pipeline


def _decode_gradio_audio(payload) -> tuple[np.ndarray, int]:  # noqa: ANN001
    """Normalise Gradio's ``(sample_rate, ndarray)`` into mono float32."""
    from .audio.wav import resample, to_mono

    if payload is None:
        return np.zeros(0, dtype=np.float32), SAMPLE_RATE
    sample_rate, audio = payload
    audio = np.asarray(audio)
    if audio.dtype.kind in "iu":
        audio = audio.astype(np.float32) / float(np.iinfo(audio.dtype).max)
    audio = to_mono(audio.astype(np.float32))
    if sample_rate != SAMPLE_RATE:
        audio = resample(audio, sample_rate, SAMPLE_RATE)
    return audio, SAMPLE_RATE


def _format_stats(result) -> str:  # noqa: ANN001
    if not result or not result.transcript:
        return "_No speech detected._"
    rtf = result.real_time_factor
    verdict = "faster than real time" if 0 < rtf < 1 else "slower than real time"
    return (
        f"**{result.source_language_name} -> {result.target_language_name}**  \n"
        f"ASR `{result.timings.asr * 1000:.0f} ms` · "
        f"MT `{result.timings.mt * 1000:.0f} ms` · "
        f"TTS `{result.timings.tts * 1000:.0f} ms` · "
        f"**total `{result.timings.total * 1000:.0f} ms`**  \n"
        f"{result.audio_duration:.1f} s of audio -> RTF `{rtf:.2f}` ({verdict})"
    )


def _write_transcript(history: list[dict]) -> str | None:
    if not history:
        return None
    lines = []
    for entry in history:
        lines.append(f"[{entry['src']}] {entry['transcript']}")
        lines.append(f"[{entry['tgt']}] {entry['translation']}")
        lines.append("")
    path = Path(tempfile.gettempdir()) / "speech_translate_transcript.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return str(path)


def _render_history(history: list[dict], subtitles_only: bool) -> str:
    if not history:
        return "_Transcript will appear here._"
    blocks = []
    for entry in reversed(history[-30:]):
        if subtitles_only:
            blocks.append(f"**{entry['translation']}**")
        else:
            blocks.append(f"**{entry['translation']}**  \n<sub>{entry['transcript']}</sub>")
    return "\n\n".join(blocks)


def _render_live(history: list[dict], partial: tuple[str, str], subtitles_only: bool) -> str:
    """The running transcript with the in-progress caption on top."""
    transcript, translation = partial
    if not translation and not transcript:
        return _render_history(history, subtitles_only)
    if subtitles_only:
        caption = f"*{translation} …*"
    else:
        caption = f"*{translation} …*  \n<sub>{transcript} …</sub>"
    if not history:
        return caption
    return f"{caption}\n\n{_render_history(history, subtitles_only)}"


def _history_entry(result) -> dict:  # noqa: ANN001
    return {
        "transcript": result.transcript,
        "translation": result.translation,
        "src": result.source_language_name,
        "tgt": result.target_language_name,
    }


def build_demo(settings: Settings | None = None, preload: bool = False):
    """Construct the Gradio Blocks app."""
    import gradio as gr

    settings = settings or Settings(src=AUTO, tgt="spa_Latn")
    cache = _ModelCache(settings)
    if preload:
        cache.pipeline(settings.tgt, speak=True)

    speakable = dict(languages_with_speech())
    target_choices = [
        (f"{name}{'' if code in speakable else '  (text only)'}", code)
        for code, name in supported_languages()
    ]
    source_choices = [("Detect automatically", AUTO)] + [
        (name, code) for code, name in supported_languages()
    ]

    # -- callbacks -------------------------------------------------------
    def translate_clip(payload, src, tgt, speak, subtitles_only, history):  # noqa: ANN001
        history = list(history or [])
        if payload is None:
            return (
                "_Record or upload something first._",
                "",
                "",
                None,
                _render_history(history, subtitles_only),
                history,
                None,
            )
        audio, _ = _decode_gradio_audio(payload)
        if audio.size == 0:
            return (
                "_Empty recording._",
                "",
                "",
                None,
                _render_history(history, subtitles_only),
                history,
                None,
            )

        pipeline = cache.pipeline(tgt, speak=speak and not subtitles_only)
        with cache.infer_lock:
            pipeline.settings.src = src
            result = pipeline.process(audio, speak=speak and not subtitles_only)

        audio_out = None
        if result.speech and len(result.speech.audio):
            audio_out = (result.speech.sample_rate, result.speech.audio)

        if result.transcript:
            history.append(_history_entry(result))
        return (
            _format_stats(result),
            result.transcript or "",
            result.translation or "",
            audio_out,
            _render_history(history, subtitles_only),
            history,
            _write_transcript(history),
        )

    # One LiveTranslator per browser session. It owns a thread, so only its
    # key goes in gr.State; the State's delete callback stops it.
    live_sessions: dict[str, LiveTranslator] = {}
    live_lock = threading.Lock()

    def live_session(key, src, tgt, speak, subtitles_only):  # noqa: ANN001
        with live_lock:
            live = live_sessions.get(key)
            if live is None:
                live = LiveTranslator(
                    lambda target, wants_speech: cache.pipeline(target, speak=wants_speech),
                    audio=settings.audio,
                    vad=settings.vad,
                    lock=cache.infer_lock,
                ).start()
                live_sessions[key] = live
        live.configure(src=src, tgt=tgt, speak=speak and not subtitles_only)
        return live

    def stream_audio(chunk, src, tgt, speak, subtitles_only, key):  # noqa: ANN001
        """Handle one streamed microphone chunk. Must return quickly."""
        key = key or uuid.uuid4().hex
        live = live_session(key, src, tgt, speak, subtitles_only)
        audio, _ = _decode_gradio_audio(chunk)
        if audio.size == 0:
            return 0.0, key
        live.feed(audio)
        level = float(np.sqrt(np.mean(np.square(audio))))
        return round(min(level * 8, 1.0), 3), key

    def stop_stream(key):  # noqa: ANN001
        live = live_sessions.get(key)
        if live is not None:
            live.end_of_stream()

    def poll_live(subtitles_only, history, key):  # noqa: ANN001
        """Timer tick: pull finished sentences, the live caption and speech."""
        live = live_sessions.get(key)
        if live is None:
            return gr.skip(), gr.skip(), gr.skip()
        update = live.poll()
        if not update.changed:
            return gr.skip(), gr.skip(), gr.skip()
        history = list(history or [])
        history.extend(_history_entry(r) for r in update.finals if r.transcript)
        rendered = _render_live(
            history, (update.partial_transcript, update.partial_translation), subtitles_only
        )
        speech = gr.skip()
        if update.speech is not None:
            speech = (update.speech.sample_rate, update.speech.audio)
        return rendered, history, speech

    def end_session(key):  # noqa: ANN001
        with live_lock:
            live = live_sessions.pop(key, None)
        if live is not None:
            live.stop()

    def clear_history(subtitles_only):  # noqa: ANN001
        return _render_history([], subtitles_only), [], None

    # -- layout ----------------------------------------------------------
    with gr.Blocks(title="Real-Time Speech Translator") as demo:
        gr.Markdown(_DESCRIPTION)

        with gr.Row():
            source_dropdown = gr.Dropdown(
                choices=source_choices, value=AUTO, label="I am speaking", scale=2
            )
            target_dropdown = gr.Dropdown(
                choices=target_choices, value=settings.tgt, label="Translate into", scale=2
            )
            speak_toggle = gr.Checkbox(value=True, label="Speak the translation", scale=1)
            subtitles_toggle = gr.Checkbox(value=False, label="Subtitles only", scale=1)

        history_state = gr.State([])

        with gr.Tab("Translate a clip"):
            gr.Markdown(
                "Record yourself or upload a file, then press **Translate**. "
                "Works without a microphone if you upload audio."
            )
            with gr.Row():
                with gr.Column():
                    clip_input = gr.Audio(
                        sources=["microphone", "upload"], type="numpy", label="Your speech"
                    )
                    translate_button = gr.Button("Translate", variant="primary")
                with gr.Column():
                    stats_output = gr.Markdown("_Latency will appear here._")
                    transcript_output = gr.Textbox(label="Heard (source language)", lines=3)
                    translation_output = gr.Textbox(label="Translation", lines=3)
                    speech_output = gr.Audio(label="Spoken translation", autoplay=True)

        with gr.Tab("Live"):
            gr.Markdown(
                "Streams your microphone. A provisional translation follows you while "
                "you talk (in italics); the final one, and its speech, arrive as each "
                "sentence ends."
            )
            with gr.Row():
                with gr.Column(scale=1):
                    tab_toggle = gr.Checkbox(
                        value=False,
                        label="Listen to another browser tab instead of the microphone",
                        info="Chrome/Edge: tick this, press Record, pick the tab and "
                        "keep 'Share tab audio' on.",
                    )
                    original_volume = gr.Slider(
                        0,
                        100,
                        value=20,
                        step=5,
                        label="Original tab volume (%)",
                        info="Lowered further while the translation speaks. Chrome/Edge only.",
                    )
                    stream_input = gr.Audio(
                        sources=["microphone"],
                        streaming=True,
                        type="numpy",
                        label="Live microphone",
                    )
                    level_meter = gr.Slider(
                        0, 1, value=0, label="Microphone level", interactive=False
                    )
                    live_speech = gr.Audio(label="Spoken translation", autoplay=True)
                with gr.Column(scale=2):
                    live_transcript = gr.Markdown("_Transcript will appear here._")
            live_timer = gr.Timer(0.25)
            live_key = gr.State("", delete_callback=end_session)

        with gr.Row():
            transcript_view = gr.Markdown("_Transcript will appear here._")
        with gr.Row():
            download_button = gr.DownloadButton("Download transcript", size="sm")
            clear_button = gr.Button("Clear transcript", size="sm")

        gr.Markdown(
            "---\n"
            f"Models: `faster-whisper {settings.asr.model}` · `{settings.mt.model}` · `Piper`  \n"
            "[Source on GitHub](https://github.com/gebrilfradj/realtime-speech-translator)"
        )

        translate_button.click(
            translate_clip,
            inputs=[
                clip_input,
                source_dropdown,
                target_dropdown,
                speak_toggle,
                subtitles_toggle,
                history_state,
            ],
            outputs=[
                stats_output,
                transcript_output,
                translation_output,
                speech_output,
                transcript_view,
                history_state,
                download_button,
            ],
        )

        stream_input.stream(
            stream_audio,
            inputs=[
                stream_input,
                source_dropdown,
                target_dropdown,
                speak_toggle,
                subtitles_toggle,
                live_key,
            ],
            outputs=[level_meter, live_key],
            show_progress="hidden",
            # Small chunks: the VAD can only end an utterance once it has
            # received the silence that ends it.
            stream_every=0.25,
            # A stream holds its slot for as long as recording lasts; with the
            # default limit of one, a second tab (or a stale one) waits forever.
            # The handler only feeds the VAD, so there is nothing to protect.
            concurrency_limit=None,
        )
        stream_input.stop_recording(stop_stream, inputs=[live_key])
        tab_toggle.change(
            None,
            inputs=[tab_toggle],
            js="(listen) => { window.__speechTranslateTab = listen; }",
        )
        original_volume.change(
            None,
            inputs=[original_volume],
            js="(v) => { window.__speechTranslateSetVolume && window.__speechTranslateSetVolume(v); }",
        )
        live_timer.tick(
            poll_live,
            inputs=[subtitles_toggle, history_state, live_key],
            outputs=[live_transcript, history_state, live_speech],
            show_progress="hidden",
            concurrency_limit=None,
        )

        clear_button.click(
            clear_history,
            inputs=[subtitles_toggle],
            outputs=[transcript_view, history_state, download_button],
        )

    return demo


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="speech-translate-web",
        description="Launch the browser demo for the speech translator.",
    )
    parser.add_argument("--host", default="127.0.0.1", help="Bind address.")
    parser.add_argument("--port", type=int, default=7860, help="Port (default: 7860).")
    parser.add_argument("--share", action="store_true", help="Create a public Gradio link.")
    parser.add_argument("--tgt", default="es", help="Initial target language.")
    parser.add_argument("--src", default=AUTO, help="Initial source language.")
    parser.add_argument("--asr-model", default="base", help="faster-whisper model.")
    parser.add_argument(
        "--mt-model", default=MTSettings.model, help="Translation model."
    )
    parser.add_argument("--device", default="auto", choices=("auto", "cpu", "cuda"))
    parser.add_argument(
        "--preload", action="store_true", help="Load models at start-up instead of on first use."
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")
    settings = Settings(
        src=args.src,
        tgt=args.tgt,
        asr=ASRSettings(model=args.asr_model, device=args.device),
        mt=MTSettings(model=args.mt_model, device=args.device),
        tts=TTSSettings(backend="auto"),
    )

    started = time.perf_counter()
    demo = build_demo(settings, preload=args.preload)
    if args.preload:
        logger.info("Models ready in %.1f s", time.perf_counter() - started)
    launch_demo(demo.queue(), server_name=args.host, server_port=args.port, share=args.share)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

import asyncio
import re
import subprocess
import uuid
from pathlib import Path
from typing import Optional

from faster_whisper import WhisperModel

import config

MAX_RETRIES = 2


class TranscriptionWorker:
    def __init__(self):
        self.model: Optional[WhisperModel] = None

    def load_model(self, progress_callback=None):
        if progress_callback is None:
            progress_callback = lambda msg: None
        progress_callback("⏳ Загрузка модели...")
        self.model = WhisperModel(
            config.WHISPER_MODEL,
            device=config.WHISPER_DEVICE,
            compute_type=config.WHISPER_COMPUTE_TYPE,
        )
        progress_callback("✅ Модель загружена")

    async def transcribe_local(
        self, input_path: str, lang: str = "en", mode: str = "raw", progress_callback=None
    ) -> tuple[str, str, Path]:
        if progress_callback is None:
            progress_callback = lambda msg: None

        if self.model is None:
            self.load_model(progress_callback)

        input_path = Path(input_path)
        output_path = input_path.with_suffix(".txt")

        task_id = str(uuid.uuid4())
        temp_dir = config.TMP_DIR / task_id
        temp_dir.mkdir(exist_ok=True)

        try:
            progress_callback("🔊 Извлечение аудио...")
            audio_path = await self._extract_audio(input_path, temp_dir)

            duration = await self._get_duration(audio_path)
            minutes = int(duration // 60)
            seconds = int(duration % 60)
            progress_callback(f"⏱ Длительность: {minutes} мин {seconds} сек")

            progress_callback("📝 Транскрибация...")
            text, detected_lang = await self._transcribe(
                audio_path,
                lang,
                lambda p: progress_callback(f"📝 Прогресс: {int(p*100)}%"),
            )

            if mode == "formatted":
                if config.OLLAMA_ENABLED:
                    progress_callback("✨ Форматирование через Ollama...")
                    text = await self._format_with_ollama(text)
                else:
                    progress_callback("📝 Форматирование текста...")
                    text = self._format_simple(text)

            output_path.write_text(text, encoding="utf-8")

            return text, detected_lang, output_path

        except Exception as e:
            progress_callback(f"❌ Ошибка: {e}")
            raise
        finally:
            await self._cleanup(temp_dir)

    def _format_simple(self, text: str) -> str:
        text = re.sub(r'\s+', ' ', text).strip()
        text = re.sub(r'(?<=[а-яА-Яa-zA-Z])\s+(?=[А-ЯA-Z])', '\n\n', text)
        sentences = re.split(r'(?<=[.!?])\s+', text)
        formatted = []
        for s in sentences:
            s = s.strip()
            if s and s[0].islower():
                s = s[0].upper() + s[1:]
            if s and not s[-1] in '.!?':
                s += '.'
            formatted.append(s)
        return '\n'.join(formatted)

    async def _transcribe(
        self, audio_path: Path, lang: str, progress_callback
    ) -> tuple[str, str]:
        total_duration = await self._get_duration(audio_path)
        chunk_duration = config.CHUNK_DURATION_SEC
        overlap = config.CHUNK_OVERLAP_SEC

        whisper_lang = lang if lang != "auto" else None

        all_text = []
        detected_language = None

        for i in range(0, int(total_duration), chunk_duration - overlap):
            start = i
            end = min(i + chunk_duration, total_duration)

            segment_path = audio_path.parent / f"chunk_{i}.wav"
            await self._extract_segment(audio_path, segment_path, start, end)

            segments, info = self.model.transcribe(
                str(segment_path),
                language=whisper_lang,
                beam_size=5,
                temperature=0.0,
                vad_filter=True,
                condition_on_previous_text=True,
            )

            if detected_language is None:
                detected_language = info.language or "unknown"

            chunk_text = " ".join([seg.text.strip() for seg in segments])
            all_text.append(chunk_text)

            segment_path.unlink()
            progress_callback((end / total_duration))

        return self._merge_chunks(all_text), detected_language

    async def _extract_audio(self, input_path: Path, output_dir: Path) -> Path:
        output_path = output_dir / "audio.wav"
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            "-y", str(output_path),
        ]
        await self._run_ffmpeg(cmd)
        return output_path

    async def _run_ffmpeg(self, args: list[str], retries: int = MAX_RETRIES):
        last_error = None
        for attempt in range(retries):
            proc = await asyncio.create_subprocess_exec(
                *args, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            stdout, stderr = await proc.communicate()
            if proc.returncode == 0:
                return
            last_error = stderr.decode("utf-8", errors="replace") if stderr else "Unknown error"
            if attempt < retries - 1:
                await asyncio.sleep(1)
        raise RuntimeError(f"FFmpeg failed after {retries} attempts: {last_error}")

    async def _get_duration(self, audio_path: Path) -> float:
        cmd = [
            "ffprobe", "-v", "error", "-show_entries", "format=duration",
            "-of", "default=noprint_wrappers=1:nokey=1", str(audio_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode == 0 and stdout:
            try:
                return float(stdout.decode().strip())
            except ValueError:
                pass
        return 0

    async def _extract_segment(
        self, input_path: Path, output_path: Path, start_sec: float, end_sec: float
    ):
        cmd = [
            "ffmpeg", "-i", str(input_path),
            "-ss", str(start_sec), "-t", str(end_sec - start_sec),
            "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
            "-y", str(output_path),
        ]
        proc = await asyncio.create_subprocess_exec(
            *cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE
        )
        stdout, stderr = await proc.communicate()
        if proc.returncode != 0:
            err = stderr.decode("utf-8", errors="replace") if stderr else "Unknown"
            raise RuntimeError(f"FFmpeg segment extraction failed: {err[:200]}")

    def _merge_chunks(self, chunks: list[str]) -> str:
        if not chunks:
            return ""
        if len(chunks) == 1:
            return self._clean_text(chunks[0])

        result = chunks[0]
        overlap_words = 5

        for chunk in chunks[1:]:
            cleaned_chunk = self._clean_text(chunk)
            if not cleaned_chunk:
                continue

            chunk_words = cleaned_chunk.split()
            result_words = result.split()

            last_words = set(result_words[-overlap_words:])
            first_keep = [w for w in chunk_words if w.lower() not in last_words]

            if first_keep:
                result += " " + " ".join(first_keep)
            else:
                result += " " + chunk_words[-1]

        return self._normalize_whitespace(result)

    def _clean_text(self, text: str) -> str:
        text = re.sub(r"\s+", " ", text)
        return text.strip()

    def _normalize_whitespace(self, text: str) -> str:
        text = re.sub(r"\n{3,}", "\n\n", text)
        text = re.sub(r" {2,}", " ", text)
        return text.strip()

    async def _format_with_ollama(self, text: str) -> str:
        import httpx

        prompt = (
            "Отформатируй текст: исправь пунктуацию, убери повторы слов, "
            "раздели на логические абзацы. Верни только текст без комментариев.\n\n"
            f"Текст:\n{text}"
        )
        try:
            async with httpx.AsyncClient() as client:
                resp = await client.post(
                    f"{config.OLLAMA_URL}/api/chat",
                    json={
                        "model": config.OLLAMA_MODEL,
                        "messages": [{"role": "user", "content": prompt}],
                        "stream": False,
                    },
                    timeout=120.0,
                )
                resp.raise_for_status()
                return resp.json()["message"]["content"]
        except Exception as e:
            import logging
            logging.warning(f"Ollama недоступна: {e}. Возвращаю сырой текст.")
            return text

    async def _cleanup(self, temp_dir: Path):
        try:
            import shutil
            shutil.rmtree(temp_dir)
        except Exception:
            pass

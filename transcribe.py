import asyncio
import argparse
from pathlib import Path

import config
from worker import TranscriptionWorker


async def main():
    parser = argparse.ArgumentParser(
        description="Локальная транскрибация видео/аудио в текст (Whisper)"
    )
    parser.add_argument("input", help="Путь к видео или аудиофайлу")
    parser.add_argument(
        "--lang", default="en", choices=["en", "ru"],
        help="Язык распознавания (по умолчанию: en)"
    )
    parser.add_argument(
        "--mode", default="raw", choices=["raw", "formatted"],
        help="Режим вывода: raw — сырой текст, formatted — с форматированием (требуется Ollama)"
    )
    parser.add_argument(
        "--output", "-o", default=None,
        help="Путь для сохранения результата (по умолчанию: рядом с исходным файлом)"
    )
    parser.add_argument(
        "--model", default=None,
        help="Модель Whisper (tiny, base, small, medium, large-v3-turbo). "
             "Переопределяет значение из .env"
    )
    parser.add_argument(
        "--device", default=None, choices=["cuda", "cpu"],
        help="Устройство. Переопределяет значение из .env"
    )

    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"❌ Файл не найден: {input_path}")
        return

    supported_ext = input_path.suffix.lower()
    if supported_ext not in config.SUPPORTED_EXTENSIONS:
        print(f"❌ Неподдерживаемый формат: {supported_ext}")
        print(f"   Поддерживаются: {', '.join(sorted(config.SUPPORTED_EXTENSIONS))}")
        return

    if args.model:
        config.WHISPER_MODEL = args.model
    if args.device:
        config.WHISPER_DEVICE = args.device

    print(f"🎬 Файл: {input_path.name}")
    print(f"🌐 Язык: {args.lang}")
    print(f"📝 Режим: {args.mode}")
    print(f"🧠 Модель: {config.WHISPER_MODEL} ({config.WHISPER_DEVICE})")
    print()

    worker = TranscriptionWorker()

    def progress(msg: str):
        print(msg)

    try:
        text, detected_lang, output_path = await worker.transcribe_local(
            str(input_path), lang=args.lang, mode=args.mode, progress_callback=progress
        )

        if args.output:
            import shutil
            final_path = Path(args.output)
            final_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(output_path), str(final_path))
            print(f"\n✅ Сохранено: {final_path}")
        else:
            print(f"\n✅ Сохранено: {output_path}")

        print(f"🌐 Распознанный язык: {detected_lang}")
        print(f"📊 Длина текста: {len(text)} символов")

    except Exception as e:
        print(f"\n❌ Ошибка: {e}")


if __name__ == "__main__":
    asyncio.run(main())

import asyncio
import json
import subprocess
import threading
from pathlib import Path
from tkinter import filedialog

import customtkinter as ctk

import config
from worker import TranscriptionWorker

SETTINGS_FILE = Path(__file__).parent / "settings.json"

DEFAULT_SETTINGS = {
    "lang": "en",
    "mode": "raw",
    "device": "cuda",
    "model": "medium",
    "output_dir": "",
    "appearance": "dark",
}


def load_settings():
    try:
        if SETTINGS_FILE.exists():
            data = json.loads(SETTINGS_FILE.read_text("utf-8"))
            return {**DEFAULT_SETTINGS, **data}
    except Exception:
        pass
    return dict(DEFAULT_SETTINGS)


def save_settings(data: dict):
    try:
        current = load_settings()
        current.update(data)
        SETTINGS_FILE.write_text(json.dumps(current, indent=2, ensure_ascii=False), "utf-8")
    except Exception:
        pass


class TranscriptionApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.settings = load_settings()
        ctk.set_appearance_mode(self.settings.get("appearance", "dark"))

        self.title("Speech2Text — транскрибация видео/аудио")
        self.geometry("800x720")
        self.minsize(680, 600)

        self.file_paths: list[str] = []
        self.result_text: str = ""
        self.result_path: Path | None = None
        self.worker: TranscriptionWorker | None = None
        self.is_processing = False
        self.saved_text: str = ""

        self._setup_ui()
        self._apply_settings()

    def _setup_ui(self):
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(4, weight=1)

        self._build_header()
        self._build_file_selector()
        self._build_settings()
        self._build_progress()
        self._build_result()
        self._build_statusbar()

    def _build_header(self):
        header = ctk.CTkFrame(self, corner_radius=0, fg_color="transparent")
        header.grid(row=0, column=0, padx=25, pady=(15, 2), sticky="ew")

        ctk.CTkLabel(
            header, text="Speech2Text", font=("Segoe UI", 26, "bold")
        ).grid(row=0, column=0, sticky="w")

        ctk.CTkLabel(
            header,
            text="Локальная транскрибация видео и аудио в текст",
            font=("Segoe UI", 12),
            text_color=("gray60", "gray70"),
        ).grid(row=1, column=0, sticky="w")

    def _build_file_selector(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=1, column=0, padx=25, pady=(8, 4), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        ctk.CTkLabel(frame, text="Файлы:", font=("Segoe UI", 12)).grid(
            row=0, column=0, padx=(10, 5), pady=10, sticky="w"
        )

        self.file_entry = ctk.CTkEntry(
            frame, placeholder_text="Выберите файлы или перетащите их сюда..."
        )
        self.file_entry.grid(row=0, column=1, padx=5, pady=10, sticky="ew")

        ctk.CTkButton(
            frame, text="Обзор", width=80, command=self._select_files
        ).grid(row=0, column=2, padx=(5, 10), pady=10)

        self.files_frame = ctk.CTkScrollableFrame(frame, height=50, fg_color="transparent")
        self.files_frame.grid(row=1, column=0, columnspan=3, padx=10, pady=(0, 6), sticky="ew")
        self.files_frame.grid_columnconfigure(0, weight=1)
        self.files_frame.grid_remove()

    def _build_settings(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=2, column=0, padx=25, pady=(4, 4), sticky="ew")
        frame.grid_columnconfigure((0, 1, 2, 3, 4), weight=1)

        combos = [
            ("Язык", ["en", "ru"], "lang_combo"),
            ("Режим", ["raw", "formatted"], "mode_combo"),
            ("Устройство", ["cuda", "cpu"], "device_combo"),
            ("Модель", ["tiny", "base", "small", "medium", "large-v3-turbo"], "model_combo"),
        ]

        for i, (label, values, attr) in enumerate(combos):
            ctk.CTkLabel(frame, text=label, font=("Segoe UI", 11)).grid(
                row=0, column=i, padx=6, pady=(8, 0), sticky="sw"
            )
            combo = ctk.CTkComboBox(frame, values=values, state="readonly", width=120)
            combo.grid(row=1, column=i, padx=6, pady=(0, 8), sticky="ew")
            setattr(self, attr, combo)

        ctk.CTkLabel(frame, text="Папка вывода", font=("Segoe UI", 11)).grid(
            row=0, column=4, padx=6, pady=(8, 0), sticky="sw"
        )
        out_frame = ctk.CTkFrame(frame, fg_color="transparent")
        out_frame.grid(row=1, column=4, padx=6, pady=(0, 8), sticky="ew")
        out_frame.grid_columnconfigure(0, weight=1)

        self.output_entry = ctk.CTkEntry(out_frame, placeholder_text="Рядом с файлом")
        self.output_entry.grid(row=0, column=0, sticky="ew")

        ctk.CTkButton(out_frame, text="...", width=30, command=self._select_output_dir).grid(
            row=0, column=1, padx=(4, 0)
        )

    def _build_progress(self):
        frame = ctk.CTkFrame(self)
        frame.grid(row=3, column=0, padx=25, pady=(4, 4), sticky="ew")
        frame.grid_columnconfigure(1, weight=1)

        self.progress_bar = ctk.CTkProgressBar(frame, mode="determinate", height=12)
        self.progress_bar.set(0)
        self.progress_bar.grid(row=0, column=0, columnspan=3, padx=10, pady=(8, 4), sticky="ew")

        self.status_label = ctk.CTkLabel(
            frame, text="Готов к работе", font=("Segoe UI", 11), anchor="w"
        )
        self.status_label.grid(row=1, column=0, padx=10, pady=(0, 8), sticky="w")

        self.file_counter = ctk.CTkLabel(
            frame, text="", font=("Segoe UI", 11), text_color=("gray50", "gray60")
        )
        self.file_counter.grid(row=1, column=1, padx=10, pady=(0, 8), sticky="e")

        self.run_btn = ctk.CTkButton(
            frame,
            text="▶  Запустить",
            font=("Segoe UI", 14, "bold"),
            height=38,
            command=self._run,
            fg_color="#2b7a4b",
            hover_color="#1f5c38",
        )
        self.run_btn.grid(row=0, column=3, rowspan=2, padx=(15, 10), pady=8)

    def _build_result(self):
        result_frame = ctk.CTkFrame(self)
        result_frame.grid(row=4, column=0, padx=25, pady=(4, 6), sticky="nsew")
        result_frame.grid_columnconfigure(0, weight=1)
        result_frame.grid_rowconfigure(1, weight=1)

        header = ctk.CTkFrame(result_frame, fg_color="transparent")
        header.grid(row=0, column=0, padx=10, pady=(4, 2), sticky="ew")
        header.grid_columnconfigure(0, weight=1)

        ctk.CTkLabel(header, text="Результат", font=("Segoe UI", 14, "bold")).grid(
            row=0, column=0, sticky="w"
        )

        btn_frame = ctk.CTkFrame(header, fg_color="transparent")
        btn_frame.grid(row=0, column=1, sticky="e")

        self.save_btn = ctk.CTkButton(
            btn_frame, text="💾 Сохранить", width=100, command=self._save, state="disabled",
            fg_color="#2b7a4b", hover_color="#1f5c38",
        )
        self.save_btn.grid(row=0, column=0, padx=2)

        self.copy_btn = ctk.CTkButton(
            btn_frame, text="📋 Копировать", width=100, command=self._copy_result, state="disabled"
        )
        self.copy_btn.grid(row=0, column=1, padx=2)

        self.saveas_btn = ctk.CTkButton(
            btn_frame, text="Сохранить как...", width=110, command=self._save_as, state="disabled"
        )
        self.saveas_btn.grid(row=0, column=2, padx=2)

        self.open_btn = ctk.CTkButton(
            btn_frame,
            text="📄 Открыть",
            width=90,
            command=self._open_result,
            state="disabled",
            fg_color="#1f5380",
            hover_color="#153b5c",
        )
        self.open_btn.grid(row=0, column=3, padx=2)

        self.result_textbox = ctk.CTkTextbox(result_frame, wrap="word", font=("Consolas", 12))
        self.result_textbox.grid(row=1, column=0, padx=10, pady=4, sticky="nsew")
        self.result_textbox.bind("<KeyRelease>", lambda e: self._on_text_change())

    def _build_statusbar(self):
        self.statusbar = ctk.CTkLabel(
            self,
            text="",
            font=("Segoe UI", 11),
            text_color=("gray50", "gray60"),
            anchor="w",
        )
        self.statusbar.grid(row=5, column=0, padx=30, pady=(0, 6), sticky="ew")

    def _apply_settings(self):
        self.lang_combo.set(self.settings.get("lang", "en"))
        self.mode_combo.set(self.settings.get("mode", "raw"))
        self.device_combo.set(self.settings.get("device", "cuda"))
        self.model_combo.set(self.settings.get("model", "medium"))
        out = self.settings.get("output_dir", "")
        if out:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, out)

    def _select_files(self):
        paths = filedialog.askopenfilenames(
            title="Выберите видео или аудиофайлы",
            filetypes=[
                ("Video/Audio", "*.mp4 *.mkv *.avi *.mov *.webm *.mp3 *.wav *.ogg *.m4a *.flac"),
                ("All files", "*.*"),
            ],
        )
        if paths:
            self.file_paths = list(paths)
            self._update_files_ui()

    def _update_files_ui(self):
        for w in self.files_frame.winfo_children():
            w.destroy()

        if not self.file_paths:
            self.files_frame.grid_remove()
            return

        self.files_frame.grid()
        self.file_entry.delete(0, "end")
        self.file_entry.insert(0, f"Выбрано файлов: {len(self.file_paths)}")

        for path in self.file_paths:
            p = Path(path)
            size_mb = p.stat().st_size / (1024 * 1024)
            row = ctk.CTkFrame(self.files_frame, fg_color="transparent")
            row.grid(sticky="ew", pady=1)
            row.grid_columnconfigure(0, weight=1)

            ctk.CTkLabel(
                row,
                text=f"  {p.name}  ({size_mb:.1f} MB)",
                font=("Segoe UI", 11),
                anchor="w",
            ).grid(row=0, column=0, sticky="ew", padx=4)

            ctk.CTkButton(
                row,
                text="✕",
                width=24,
                fg_color="transparent",
                hover_color=("gray80", "gray30"),
                command=lambda fp=path: self._remove_file(fp),
            ).grid(row=0, column=1)

    def _remove_file(self, path: str):
        self.file_paths = [p for p in self.file_paths if p != path]
        self._update_files_ui()

    def _select_output_dir(self):
        path = filedialog.askdirectory(title="Папка для сохранения результатов")
        if path:
            self.output_entry.delete(0, "end")
            self.output_entry.insert(0, path)

    def _log(self, text: str):
        self.status_label.configure(text=text)

    def _set_ui_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.run_btn.configure(state=state)
        self.lang_combo.configure(state=state)
        self.mode_combo.configure(state=state)
        self.device_combo.configure(state=state)
        self.model_combo.configure(state=state)

    def _run(self):
        if not self.file_paths:
            self._log("❌ Сначала выберите файлы")
            return

        if self.is_processing:
            return

        self.is_processing = True
        self.result_text = ""
        self.result_path = None
        self.result_textbox.delete("0.0", "end")
        self.copy_btn.configure(state="disabled")
        self.saveas_btn.configure(state="disabled")
        self.open_btn.configure(state="disabled")
        self.progress_bar.set(0)

        config.WHISPER_MODEL = self.model_combo.get()
        config.WHISPER_DEVICE = self.device_combo.get()

        save_settings({
            "lang": self.lang_combo.get(),
            "mode": self.mode_combo.get(),
            "device": self.device_combo.get(),
            "model": self.model_combo.get(),
            "output_dir": self.output_entry.get(),
        })

        threading.Thread(target=self._run_async, daemon=True).start()

    def _run_async(self):
        try:
            self.worker = TranscriptionWorker()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

            total = len(self.file_paths)
            combined_text = ""

            for idx, file_path in enumerate(self.file_paths, 1):
                if not self.is_processing:
                    break

                fname = Path(file_path).name
                self.after(0, lambda idx=idx, total=total, fname=fname: (
                    self.file_counter.configure(text=f"Файл {idx}/{total}: {fname}")
                ))

                text, detected_lang, output_path = loop.run_until_complete(
                    self.worker.transcribe_local(
                        file_path,
                        lang=self.lang_combo.get(),
                        mode=self.mode_combo.get(),
                        progress_callback=lambda msg: self.after(0, lambda m=msg: self._log(m)),
                    )
                )

                output_dir = self.output_entry.get().strip()
                if output_dir:
                    import shutil
                    final_path = Path(output_dir) / f"{Path(file_path).stem}.txt"
                    final_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.move(str(output_path), str(final_path))
                    self.result_path = final_path
                else:
                    self.result_path = output_path

                combined_text += f"--- {fname} ---\n{text}\n\n"
                self.result_text = combined_text

                pct = idx / total
                self.after(0, lambda p=pct: self.progress_bar.set(p))

            self.after(0, self._show_result)
            self.after(0, lambda: self.statusbar.configure(text=f"✅ Обработано файлов: {total}"))

        except Exception as e:
            err_msg = str(e)
            self.after(0, lambda m=err_msg: self._log(f"❌ Ошибка: {m}"))
        finally:
            self.after(0, lambda: self._set_ui_enabled(True))
            self.after(0, lambda: self.file_counter.configure(text=""))
            self.is_processing = False

    def _show_result(self):
        if not self.result_text:
            return
        self.saved_text = self.result_text
        self.result_textbox.delete("0.0", "end")
        self.result_textbox.insert("0.0", self.result_text)
        self.copy_btn.configure(state="normal")
        self.saveas_btn.configure(state="normal")
        self.save_btn.configure(state="disabled")
        if self.result_path:
            self.open_btn.configure(state="normal")
        self._log("✅ Готово!")

    def _copy_result(self):
        self.clipboard_clear()
        self.clipboard_append(self.result_text)
        self._log("📋 Скопировано в буфер обмена")

    def _on_text_change(self):
        current = self.result_textbox.get("0.0", "end").strip()
        if current != self.saved_text:
            self.save_btn.configure(state="normal")
        else:
            self.save_btn.configure(state="disabled")

    def _save(self):
        text = self.result_textbox.get("0.0", "end").strip()
        if self.result_path:
            self.result_path.write_text(text, encoding="utf-8")
            self.saved_text = text
            self.save_btn.configure(state="disabled")
            self._log(f"💾 Сохранено: {self.result_path.name}")

    def _save_as(self):
        text = self.result_textbox.get("0.0", "end").strip()
        path = filedialog.asksaveasfilename(
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt")],
            title="Сохранить как",
        )
        if path:
            Path(path).write_text(text, encoding="utf-8")
            self._log(f"💾 Сохранено: {path}")

    def _open_result(self):
        if self.result_path and self.result_path.exists():
            subprocess.Popen(["notepad.exe", str(self.result_path)])
        else:
            self._log("❌ Файл результата не найден")


if __name__ == "__main__":
    app = TranscriptionApp()
    app.mainloop()

from __future__ import annotations

import os
import queue
import subprocess
import sys
import threading
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from engine import get_hardware_summary, run_video


APP_NAME = "TRIBE v2 Local Runner"


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("860x620")
        self.minsize(760, 560)
        self.q: queue.Queue = queue.Queue()
        self.video = tk.StringVar()
        self.output = tk.StringVar(value=str(Path.home() / "TRIBEv2 Results"))
        self.status = tk.StringVar(value="Ready.")
        self.progress = tk.DoubleVar(value=0.0)
        self._build()
        self.after(100, self._drain)

    def _build(self):
        root = ttk.Frame(self, padding=28)
        root.pack(fill="both", expand=True)

        ttk.Label(root, text="TRIBE v2", font=("Segoe UI", 26, "bold")).pack(anchor="w")
        ttk.Label(
            root,
            text="Local cortical-response prediction for video",
            font=("Segoe UI", 12),
        ).pack(anchor="w", pady=(0, 24))

        info = ttk.LabelFrame(root, text="Hardware", padding=14)
        info.pack(fill="x", pady=(0, 18))
        try:
            hw = get_hardware_summary()
            gpu = hw["gpu"] or "CPU mode"
            text = f'{hw["logical_cpus"]} logical CPU threads • {gpu} • Torch {hw["torch"]}'
        except Exception as exc:
            text = f"Hardware check unavailable: {exc}"
        ttk.Label(info, text=text).pack(anchor="w")

        video_box = ttk.LabelFrame(root, text="1. Video", padding=14)
        video_box.pack(fill="x", pady=(0, 14))
        row = ttk.Frame(video_box)
        row.pack(fill="x")
        ttk.Entry(row, textvariable=self.video).pack(side="left", fill="x", expand=True)
        ttk.Button(row, text="Choose…", command=self._choose_video).pack(side="left", padx=(10, 0))

        out_box = ttk.LabelFrame(root, text="2. Output folder", padding=14)
        out_box.pack(fill="x", pady=(0, 14))
        row2 = ttk.Frame(out_box)
        row2.pack(fill="x")
        ttk.Entry(row2, textvariable=self.output).pack(side="left", fill="x", expand=True)
        ttk.Button(row2, text="Choose…", command=self._choose_output).pack(side="left", padx=(10, 0))

        run_box = ttk.LabelFrame(root, text="3. Process", padding=14)
        run_box.pack(fill="x", pady=(0, 14))
        self.run_button = ttk.Button(run_box, text="Analyze video locally", command=self._start)
        self.run_button.pack(anchor="w")
        ttk.Progressbar(run_box, variable=self.progress, maximum=100).pack(fill="x", pady=(14, 8))
        ttk.Label(run_box, textvariable=self.status).pack(anchor="w")

        ttk.Label(
            root,
            text=(
                "First run downloads the model and dependencies into your user profile. "
                "Inference stays on this PC after the required files are cached.\n"
                "TRIBE v2 is CC BY-NC 4.0: commercial use requires separate permission."
            ),
            wraplength=780,
        ).pack(anchor="w", pady=(10, 0))

    def _choose_video(self):
        p = filedialog.askopenfilename(
            title="Choose a video",
            filetypes=[
                ("Video files", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self.video.set(p)

    def _choose_output(self):
        p = filedialog.askdirectory(title="Choose output folder")
        if p:
            self.output.set(p)

    def _start(self):
        video = Path(self.video.get())
        if not video.is_file():
            messagebox.showerror(APP_NAME, "Choose a valid video first.")
            return
        out = Path(self.output.get())
        out.mkdir(parents=True, exist_ok=True)
        self.run_button.configure(state="disabled")
        self.progress.set(0)
        self.status.set("Starting…")
        threading.Thread(target=self._worker, args=(video, out), daemon=True).start()

    def _worker(self, video: Path, out: Path):
        try:
            def cb(msg, value):
                self.q.put(("progress", msg, value))
            result = run_video(video, out, cb)
            self.q.put(("done", result))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                if item[0] == "progress":
                    _, msg, value = item
                    self.status.set(msg)
                    if value is not None:
                        self.progress.set(float(value) * 100)
                elif item[0] == "done":
                    result = item[1]
                    self.progress.set(100)
                    self.status.set(
                        f"Done — {result.n_timesteps} timepoints × "
                        f"{result.n_vertices} cortical vertices."
                    )
                    self.run_button.configure(state="normal")
                    report = result.output_dir / "report.html"
                    webbrowser.open(report.as_uri())
                    messagebox.showinfo(APP_NAME, f"Finished.\n\n{result.output_dir}")
                elif item[0] == "error":
                    self.run_button.configure(state="normal")
                    self.status.set("Failed.")
                    error = item[1]
                    log = Path(self.output.get()) / "tribev2_last_error.txt"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(error, encoding="utf-8")
                    messagebox.showerror(
                        APP_NAME,
                        "Processing failed. A diagnostic log was saved to:\n" + str(log),
                    )
        except queue.Empty:
            pass
        self.after(100, self._drain)


if __name__ == "__main__":
    App().mainloop()

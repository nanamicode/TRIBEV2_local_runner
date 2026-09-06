from __future__ import annotations

import contextlib
import queue
import re
import sys
import threading
import time
import traceback
import webbrowser
from pathlib import Path
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from tkinter.scrolledtext import ScrolledText

from engine import get_cached_run_status, get_hardware_summary, run_video
from calibration import CONTEXT_FIELDS, MIN_TRAIN_SAMPLES, TARGET_FIELDS, save_campaign_metrics


APP_NAME = "TRIBE v2 Local Runner"
ACCENT = "#7c3aed"
BG = "#0b0d12"
PANEL = "#141821"
PANEL_2 = "#10131a"
BORDER = "#262b38"
TEXT = "#f5f7fb"
MUTED = "#9da5b4"
GOOD = "#22c55e"
WARN = "#f59e0b"
BAD = "#ef4444"


STAGES = [
    (
        "prepare",
        "01  Preparação",
        "Valida hardware, ambiente local e dispositivo de inferência.",
    ),
    (
        "model_download",
        "02  Modelos",
        "Confere o cache e baixa apenas os arquivos que ainda não existem.",
    ),
    (
        "model_load",
        "03  Rede cortical",
        "Carrega o TRIBE v2 e os pesos necessários para a previsão cerebral.",
    ),
    (
        "video_prepare",
        "04  Estímulo",
        "Lê duração, formato e organiza o vídeo na linha do tempo do modelo.",
    ),
    (
        "predict",
        "05  V-JEPA2 + TRIBE",
        "Transforma o vídeo em features visuais e prevê resposta cortical.",
    ),
    (
        "save_raw",
        "06  Dados brutos",
        "Salva a matriz cortical completa antes de qualquer resumo.",
    ),
    (
        "normalize",
        "07  Normalização",
        "Converte ~20 mil vértices em uma assinatura compacta e legível por IA.",
    ),
    (
        "visualize",
        "08  Visualização",
        "Renderiza cérebro, picos temporais, regiões e relatório interativo.",
    ),
    (
        "done",
        "09  Concluído",
        "Entrega dados brutos, pacote para ChatGPT e relatório visual.",
    ),
]


class _TeeToQueue:
    """Mirror worker stdout/stderr to the terminal and to the UI log."""

    def __init__(self, original, event_queue: queue.Queue, channel: str):
        self.original = original
        self.event_queue = event_queue
        self.channel = channel
        self._buffer = ""

    def write(self, data):
        if not data:
            return 0
        if self.original is not None:
            try:
                self.original.write(data)
                self.original.flush()
            except Exception:
                pass

        self._buffer += str(data)
        # tqdm uses carriage returns, regular logs use newlines.
        parts = re.split(r"[\r\n]+", self._buffer)
        self._buffer = parts.pop() if parts else ""
        for part in parts:
            part = part.strip()
            if part:
                self.event_queue.put(("log", self.channel, part))
        return len(data)

    def flush(self):
        if self.original is not None:
            try:
                self.original.flush()
            except Exception:
                pass


class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title(APP_NAME)
        self.geometry("1180x820")
        self.minsize(1000, 700)
        self.configure(bg=BG)

        self.q: queue.Queue = queue.Queue()
        self.video = tk.StringVar()
        self.output = tk.StringVar(value=str(Path.home() / "TRIBEv2 Results"))
        self.status = tk.StringVar(value="Pronto para analisar")
        self.objective = tk.StringVar(value="Selecione um vídeo para iniciar.")
        self.overall_progress = tk.DoubleVar(value=0.0)
        self.phase_progress = tk.DoubleVar(value=0.0)
        self.phase_label = tk.StringVar(value="Aguardando")
        self.elapsed = tk.StringVar(value="00:00")
        self.cache_hint = tk.StringVar(value="Selecione um vídeo para verificar o cache neural.")
        self._started_at: float | None = None
        self._current_stage: str | None = None
        self._last_report: Path | None = None
        self._last_run_dir: Path | None = None
        self._stage_widgets: dict[str, dict[str, tk.Widget]] = {}

        self._configure_styles()
        self._build()
        self.after(100, self._drain)
        self.after(1000, self._tick)

    def _configure_styles(self):
        style = ttk.Style(self)
        try:
            style.theme_use("clam")
        except Exception:
            pass

        style.configure(
            "Purple.Horizontal.TProgressbar",
            troughcolor="#1d2230",
            background=ACCENT,
            bordercolor="#1d2230",
            lightcolor=ACCENT,
            darkcolor=ACCENT,
            thickness=10,
        )
        style.configure(
            "Thin.Horizontal.TProgressbar",
            troughcolor="#1b202c",
            background="#9b7cff",
            bordercolor="#1b202c",
            lightcolor="#9b7cff",
            darkcolor="#9b7cff",
            thickness=6,
        )
        style.configure(
            "Dark.TEntry",
            fieldbackground="#0d1016",
            foreground=TEXT,
            insertcolor=TEXT,
            bordercolor=BORDER,
            lightcolor=BORDER,
            darkcolor=BORDER,
            padding=8,
        )

    def _card(self, parent, **pack_kwargs):
        frame = tk.Frame(
            parent,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
            bd=0,
        )
        frame.pack(**pack_kwargs)
        return frame

    def _build(self):
        top = tk.Frame(self, bg=BG)
        top.pack(fill="x", padx=28, pady=(24, 14))

        header_left = tk.Frame(top, bg=BG)
        header_left.pack(side="left", fill="x", expand=True)
        tk.Label(
            header_left,
            text="TRIBE v2",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 30, "bold"),
        ).pack(anchor="w")
        tk.Label(
            header_left,
            text="Predição cortical local para vídeo • processamento 100% neste PC",
            bg=BG,
            fg=MUTED,
            font=("Segoe UI", 11),
        ).pack(anchor="w", pady=(2, 0))

        badge = tk.Label(
            top,
            textvariable=self.status,
            bg="#201634",
            fg="#d8c8ff",
            padx=14,
            pady=8,
            font=("Segoe UI", 10, "bold"),
        )
        badge.pack(side="right", anchor="n", pady=4)

        input_card = self._card(self, fill="x", padx=28, pady=(0, 14))
        inner = tk.Frame(input_card, bg=PANEL)
        inner.pack(fill="x", padx=18, pady=16)

        try:
            hw = get_hardware_summary()
            gpu = hw["gpu"] or "CPU"
            hw_text = (
                f'{hw["logical_cpus"]} threads lógicas   •   {gpu}   •   '
                f'Torch {hw["torch"]}   •   dispositivo {hw["device"].upper()}'
            )
        except Exception as exc:
            hw_text = f"Hardware indisponível: {exc}"

        tk.Label(
            inner,
            text="HARDWARE LOCAL",
            bg=PANEL,
            fg="#a78bfa",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, columnspan=3, sticky="w")
        tk.Label(
            inner,
            text=hw_text,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 10),
        ).grid(row=1, column=0, columnspan=3, sticky="w", pady=(4, 14))

        tk.Label(inner, text="Vídeo", bg=PANEL, fg=MUTED).grid(
            row=2, column=0, sticky="w", padx=(0, 12)
        )
        ttk.Entry(inner, textvariable=self.video, style="Dark.TEntry").grid(
            row=2, column=1, sticky="ew"
        )
        tk.Button(
            inner,
            text="Selecionar",
            command=self._choose_video,
            bg="#242938",
            fg=TEXT,
            activebackground="#30364a",
            activeforeground=TEXT,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        ).grid(row=2, column=2, padx=(10, 0))

        tk.Label(inner, text="Resultados", bg=PANEL, fg=MUTED).grid(
            row=3, column=0, sticky="w", padx=(0, 12), pady=(10, 0)
        )
        ttk.Entry(inner, textvariable=self.output, style="Dark.TEntry").grid(
            row=3, column=1, sticky="ew", pady=(10, 0)
        )
        tk.Button(
            inner,
            text="Selecionar",
            command=self._choose_output,
            bg="#242938",
            fg=TEXT,
            activebackground="#30364a",
            activeforeground=TEXT,
            relief="flat",
            padx=16,
            pady=8,
            cursor="hand2",
        ).grid(row=3, column=2, padx=(10, 0), pady=(10, 0))
        tk.Label(
            inner,
            textvariable=self.cache_hint,
            bg=PANEL,
            fg="#86efac",
            font=("Segoe UI", 8),
            anchor="w",
        ).grid(row=4, column=1, columnspan=2, sticky="w", pady=(8, 0))
        inner.columnconfigure(1, weight=1)

        control = tk.Frame(self, bg=BG)
        control.pack(fill="x", padx=28, pady=(0, 14))

        self.run_button = tk.Button(
            control,
            text="ANALISAR VÍDEO",
            command=self._start,
            bg=ACCENT,
            fg="white",
            activebackground="#6d28d9",
            activeforeground="white",
            relief="flat",
            padx=22,
            pady=12,
            font=("Segoe UI", 10, "bold"),
            cursor="hand2",
        )
        self.run_button.pack(side="left")

        self.open_button = tk.Button(
            control,
            text="Abrir último relatório",
            command=self._open_report,
            bg="#202431",
            fg=TEXT,
            activebackground="#2a3040",
            activeforeground=TEXT,
            relief="flat",
            padx=18,
            pady=12,
            state="disabled",
            cursor="hand2",
        )
        self.open_button.pack(side="left", padx=(10, 0))

        self.calibration_button = tk.Button(
            control,
            text="DADOS REAIS / CALIBRAÇÃO",
            command=self._open_calibration_dialog,
            bg="#202431",
            fg=TEXT,
            activebackground="#2a3040",
            activeforeground=TEXT,
            relief="flat",
            padx=18,
            pady=12,
            state="disabled",
            cursor="hand2",
        )
        self.calibration_button.pack(side="left", padx=(10, 0))

        tk.Label(
            control,
            textvariable=self.elapsed,
            bg=BG,
            fg=MUTED,
            font=("Consolas", 10),
        ).pack(side="right")

        progress_card = self._card(self, fill="x", padx=28, pady=(0, 14))
        p = tk.Frame(progress_card, bg=PANEL)
        p.pack(fill="x", padx=18, pady=14)

        row = tk.Frame(p, bg=PANEL)
        row.pack(fill="x")
        tk.Label(
            row,
            textvariable=self.phase_label,
            bg=PANEL,
            fg=TEXT,
            font=("Segoe UI", 11, "bold"),
        ).pack(side="left")
        tk.Label(
            row,
            textvariable=self.objective,
            bg=PANEL,
            fg=MUTED,
            font=("Segoe UI", 9),
        ).pack(side="right")

        ttk.Progressbar(
            p,
            variable=self.overall_progress,
            maximum=100,
            style="Purple.Horizontal.TProgressbar",
        ).pack(fill="x", pady=(10, 8))
        ttk.Progressbar(
            p,
            variable=self.phase_progress,
            maximum=100,
            style="Thin.Horizontal.TProgressbar",
        ).pack(fill="x")

        body = tk.Frame(self, bg=BG)
        body.pack(fill="both", expand=True, padx=28, pady=(0, 18))
        body.columnconfigure(0, weight=0, minsize=340)
        body.columnconfigure(1, weight=1)
        body.rowconfigure(0, weight=1)

        stage_card = tk.Frame(
            body,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        stage_card.grid(row=0, column=0, sticky="nsew", padx=(0, 7))

        tk.Label(
            stage_card,
            text="PIPELINE",
            bg=PANEL,
            fg="#a78bfa",
            font=("Segoe UI", 9, "bold"),
        ).pack(anchor="w", padx=18, pady=(16, 10))

        for key, title, desc in STAGES:
            row = tk.Frame(stage_card, bg=PANEL)
            row.pack(fill="x", padx=16, pady=4)
            dot = tk.Label(
                row,
                text="○",
                bg=PANEL,
                fg="#566071",
                font=("Segoe UI Symbol", 13, "bold"),
                width=2,
            )
            dot.pack(side="left", anchor="n", pady=1)
            txt = tk.Frame(row, bg=PANEL)
            txt.pack(side="left", fill="x", expand=True)
            title_lbl = tk.Label(
                txt,
                text=title,
                bg=PANEL,
                fg="#c4cad4",
                font=("Segoe UI", 9, "bold"),
            )
            title_lbl.pack(anchor="w")
            desc_lbl = tk.Label(
                txt,
                text=desc,
                bg=PANEL,
                fg="#697386",
                justify="left",
                wraplength=275,
                font=("Segoe UI", 8),
            )
            desc_lbl.pack(anchor="w", pady=(1, 4))
            self._stage_widgets[key] = {
                "dot": dot,
                "title": title_lbl,
                "desc": desc_lbl,
            }

        log_card = tk.Frame(
            body,
            bg=PANEL,
            highlightbackground=BORDER,
            highlightthickness=1,
        )
        log_card.grid(row=0, column=1, sticky="nsew", padx=(7, 0))

        log_head = tk.Frame(log_card, bg=PANEL)
        log_head.pack(fill="x", padx=18, pady=(14, 8))
        tk.Label(
            log_head,
            text="ATIVIDADE EM TEMPO REAL",
            bg=PANEL,
            fg="#a78bfa",
            font=("Segoe UI", 9, "bold"),
        ).pack(side="left")
        tk.Label(
            log_head,
            text="terminal + progresso interno",
            bg=PANEL,
            fg="#667085",
            font=("Segoe UI", 8),
        ).pack(side="right")

        self.log_text = ScrolledText(
            log_card,
            bg="#090b10",
            fg="#cbd5e1",
            insertbackground=TEXT,
            selectbackground="#3b2a66",
            relief="flat",
            borderwidth=0,
            font=("Consolas", 9),
            wrap="word",
            padx=12,
            pady=10,
        )
        self.log_text.pack(fill="both", expand=True, padx=14, pady=(0, 14))
        self.log_text.configure(state="disabled")
        self.log_text.tag_configure("warn", foreground="#fbbf24")
        self.log_text.tag_configure("error", foreground="#f87171")
        self.log_text.tag_configure("good", foreground="#86efac")
        self.log_text.tag_configure("muted", foreground="#7b8495")

        footer = tk.Frame(self, bg=BG)
        footer.pack(fill="x", padx=28, pady=(0, 18))
        tk.Label(
            footer,
            text=(
                "Saída científica: previsão cortical fMRI-like de sujeito médio. "
                "Não é leitura do cérebro de uma pessoa real."
            ),
            bg=BG,
            fg="#697386",
            font=("Segoe UI", 8),
        ).pack(side="left")
        tk.Label(
            footer,
            text="TRIBE v2 • CC BY-NC 4.0",
            bg=BG,
            fg="#697386",
            font=("Segoe UI", 8),
        ).pack(side="right")

    def _choose_video(self):
        p = filedialog.askopenfilename(
            title="Escolha um vídeo",
            filetypes=[
                ("Vídeos", "*.mp4 *.mov *.mkv *.avi *.webm"),
                ("Todos os arquivos", "*.*"),
            ],
        )
        if p:
            self.video.set(p)
            self._refresh_cache_status()

    def _choose_output(self):
        p = filedialog.askdirectory(title="Escolha a pasta de resultados")
        if p:
            self.output.set(p)
            self._refresh_cache_status()

    def _refresh_cache_status(self):
        video = Path(self.video.get())
        out = Path(self.output.get())
        if not video.is_file():
            self.cache_hint.set("Selecione um vídeo para verificar o cache neural.")
            self.run_button.configure(text="ANALISAR VÍDEO")
            return
        status = get_cached_run_status(video, out)
        if status.get("raw_complete"):
            self.cache_hint.set(
                "CACHE NEURAL COMPLETO • V-JEPA2/TRIBE serão pulados; só pós-processamento será refeito."
            )
            self.run_button.configure(text="REPROCESSAR CACHE")
            run_dir = status.get("run_dir")
            if run_dir:
                self._last_run_dir = Path(run_dir)
                report = self._last_run_dir / "report.html"
                if report.exists():
                    self._last_report = report
                    self.open_button.configure(state="normal")
                self.calibration_button.configure(state="normal")
        elif status.get("exists"):
            self.cache_hint.set(
                f"Execução parcial encontrada • retomada automática a partir de {status.get('stage') or 'checkpoint'}."
            )
            self.run_button.configure(text="RETOMAR ANÁLISE")
        else:
            self.cache_hint.set("Sem cache neural completo • este vídeo exigirá inferência V-JEPA2/TRIBE.")
            self.run_button.configure(text="ANALISAR VÍDEO")

    def _start(self):
        video = Path(self.video.get())
        if not video.is_file():
            messagebox.showerror(APP_NAME, "Escolha um arquivo de vídeo válido.")
            return

        out = Path(self.output.get())
        out.mkdir(parents=True, exist_ok=True)

        self.run_button.configure(state="disabled")
        self.open_button.configure(state="disabled")
        self.calibration_button.configure(state="disabled")
        self.overall_progress.set(0)
        self.phase_progress.set(0)
        self.status.set("Processando")
        self.phase_label.set("Iniciando análise")
        self.objective.set("Preparando o pipeline local.")
        self._started_at = time.monotonic()
        self._last_report = None
        self._set_stage("prepare")
        self._clear_log()
        self._append_log("Análise iniciada. O processamento neural pesado permanece local.", "good")

        threading.Thread(target=self._worker, args=(video, out), daemon=True).start()

    def _worker(self, video: Path, out: Path):
        stdout_tee = _TeeToQueue(sys.stdout, self.q, "stdout")
        stderr_tee = _TeeToQueue(sys.stderr, self.q, "stderr")
        try:
            def cb(msg, value, stage=None):
                self.q.put(("progress", msg, value, stage))

            with contextlib.redirect_stdout(stdout_tee), contextlib.redirect_stderr(stderr_tee):
                result = run_video(video, out, cb)
            self.q.put(("done", result))
        except Exception:
            self.q.put(("error", traceback.format_exc()))

    def _set_stage(self, stage: str | None):
        if not stage:
            return
        keys = [x[0] for x in STAGES]
        if stage not in keys:
            return
        if self._current_stage == stage:
            return
        self._current_stage = stage
        current_index = keys.index(stage)

        for idx, key in enumerate(keys):
            widgets = self._stage_widgets[key]
            if idx < current_index:
                widgets["dot"].configure(text="●", fg=GOOD)
                widgets["title"].configure(fg="#d5dae3")
                widgets["desc"].configure(fg="#657084")
            elif idx == current_index:
                widgets["dot"].configure(text="●", fg="#a78bfa")
                widgets["title"].configure(fg="#d8c8ff")
                widgets["desc"].configure(fg="#aeb6c5")
            else:
                widgets["dot"].configure(text="○", fg="#4d5666")
                widgets["title"].configure(fg="#8d96a7")
                widgets["desc"].configure(fg="#596273")

        stage_def = next(x for x in STAGES if x[0] == stage)
        self.phase_label.set(stage_def[1])
        self.objective.set(stage_def[2])
        self.phase_progress.set(0)

    def _append_log(self, line: str, tag: str | None = None):
        if not line:
            return
        lower = line.lower()
        if tag is None:
            if "traceback" in lower or "error" in lower or "failed" in lower:
                tag = "error"
            elif "warning" in lower or "warn" in lower:
                tag = "warn"
            elif "100%" in line or "complete" in lower or "loaded" in lower:
                tag = "good"
            else:
                tag = "muted"

        self.log_text.configure(state="normal")
        stamp = time.strftime("%H:%M:%S")
        self.log_text.insert("end", f"[{stamp}] {line}\n", tag)

        # Keep the UI responsive during very verbose model downloads.
        try:
            lines = int(self.log_text.index("end-1c").split(".")[0])
            if lines > 900:
                self.log_text.delete("1.0", "150.0")
        except Exception:
            pass

        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _clear_log(self):
        self.log_text.configure(state="normal")
        self.log_text.delete("1.0", "end")
        self.log_text.configure(state="disabled")

    def _interpret_runtime_log(self, line: str):
        # Examples: "Encoding video: 64%|..." or "model.safetensors: 54%|..."
        matches = list(re.finditer(r"([^\r\n:]{2,45}):\s*(\d{1,3})%\|", line))
        if matches:
            label = matches[-1].group(1).strip()
            pct = min(100, max(0, int(matches[-1].group(2))))
            self.phase_progress.set(pct)
            if label:
                self.phase_label.set(label)

        lower = line.lower()
        if "preparing extractor: video" in lower:
            self._set_stage("predict")
            self.phase_label.set("Preparando V-JEPA2")
        elif "loading weights" in lower:
            self._set_stage("predict")
            self.phase_label.set("Carregando pesos visuais")
        elif "encoding video" in lower:
            self._set_stage("predict")
            self.phase_label.set("Codificando vídeo com V-JEPA2")
        elif "loaded video" in lower:
            self._set_stage("predict")
            self.phase_label.set("Vídeo decodificado")
        elif "predicted " in lower and "segments" in lower:
            self._set_stage("save_raw")
            self.phase_label.set("Previsão cortical concluída")

    def _drain(self):
        try:
            while True:
                item = self.q.get_nowait()
                kind = item[0]
                if kind == "progress":
                    _, msg, value, stage = item
                    if stage:
                        self._set_stage(stage)
                    if value is not None:
                        self.overall_progress.set(float(value) * 100)
                    self._append_log(msg, "good" if stage == "done" else None)

                elif kind == "log":
                    _, channel, line = item
                    self._interpret_runtime_log(line)
                    self._append_log(line, "error" if channel == "stderr" and "error" in line.lower() else None)

                elif kind == "done":
                    result = item[1]
                    self.overall_progress.set(100)
                    self.phase_progress.set(100)
                    self._set_stage("done")
                    self.status.set(
                        f"Concluído • {result.n_timesteps} janelas × {result.n_vertices} vértices"
                    )
                    self.phase_label.set("Análise concluída")
                    self.objective.set(
                        "Relatório visual e pacote normalizado para IA estão prontos."
                    )
                    self.run_button.configure(state="normal")
                    self._last_report = result.report_path
                    self._last_run_dir = result.output_dir
                    self.open_button.configure(state="normal")
                    self.calibration_button.configure(state="normal")
                    self._refresh_cache_status()
                    self._append_log(
                        f"Arquivos prontos em: {result.output_dir}",
                        "good",
                    )
                    try:
                        webbrowser.open(result.report_path.as_uri())
                    except Exception:
                        pass
                    messagebox.showinfo(
                        APP_NAME,
                        (
                            "Análise concluída.\n\n"
                            f"Relatório: {result.report_path}\n"
                            f"Pacote para IA: {result.normalized_path}"
                        ),
                    )

                elif kind == "error":
                    self.run_button.configure(state="normal")
                    self.status.set("Falha")
                    self.phase_label.set("Processamento interrompido")
                    error = item[1]
                    log = Path(self.output.get()) / "tribev2_last_error.txt"
                    log.parent.mkdir(parents=True, exist_ok=True)
                    log.write_text(error, encoding="utf-8")
                    self._append_log(error, "error")
                    messagebox.showerror(
                        APP_NAME,
                        "Falha no processamento. Log diagnóstico salvo em:\n" + str(log),
                    )
        except queue.Empty:
            pass
        self.after(100, self._drain)

    def _tick(self):
        if self._started_at is not None and self.run_button["state"] == "disabled":
            seconds = int(time.monotonic() - self._started_at)
            self.elapsed.set(f"{seconds // 60:02d}:{seconds % 60:02d}")
        self.after(1000, self._tick)

    def _open_calibration_dialog(self):
        run_dir = self._last_run_dir
        if run_dir is None or not run_dir.exists():
            messagebox.showerror(
                APP_NAME,
                "Nenhuma análise concluída foi encontrada para receber métricas reais.",
            )
            return

        existing = {}
        metrics_path = run_dir / "campaign_metrics.json"
        if metrics_path.exists():
            try:
                import json
                existing = json.loads(metrics_path.read_text(encoding="utf-8"))
            except Exception:
                existing = {}

        win = tk.Toplevel(self)
        win.title("TRIBE v2 • Dados reais / Calibração")
        win.geometry("760x760")
        win.minsize(680, 620)
        win.configure(bg=BG)
        win.transient(self)

        canvas = tk.Canvas(win, bg=BG, highlightthickness=0)
        scrollbar = ttk.Scrollbar(win, orient="vertical", command=canvas.yview)
        frame = tk.Frame(canvas, bg=BG)
        frame.bind(
            "<Configure>",
            lambda e: canvas.configure(scrollregion=canvas.bbox("all")),
        )
        canvas.create_window((0, 0), window=frame, anchor="nw", width=720)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")

        tk.Label(
            frame,
            text="Calibração com performance real",
            bg=BG,
            fg=TEXT,
            font=("Segoe UI", 22, "bold"),
        ).pack(anchor="w", padx=24, pady=(24, 4))
        tk.Label(
            frame,
            text=(
                "Preencha somente números medidos para ESTE criativo. "
                "Campos vazios são ignorados. O sistema só treina previsores após "
                f"{MIN_TRAIN_SAMPLES} criativos rotulados e marca 12–29 amostras como experimental."
            ),
            bg=BG,
            fg=MUTED,
            justify="left",
            wraplength=650,
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=24, pady=(0, 18))

        target_vars = {}
        context_vars = {}
        existing_targets = existing.get("targets") or {}
        existing_context = existing.get("context") or {}

        section = tk.Frame(frame, bg=PANEL, highlightbackground=BORDER, highlightthickness=1)
        section.pack(fill="x", padx=24, pady=(0, 14))
        tk.Label(
            section,
            text="MÉTRICAS DE RESULTADO",
            bg=PANEL,
            fg="#a78bfa",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 10))

        for row_idx, (key, label) in enumerate(TARGET_FIELDS.items(), start=1):
            var = tk.StringVar()
            if existing_targets.get(key) is not None:
                var.set(str(existing_targets.get(key)))
            target_vars[key] = var
            tk.Label(section, text=label, bg=PANEL, fg=MUTED).grid(
                row=row_idx, column=0, sticky="w", padx=16, pady=5
            )
            ttk.Entry(section, textvariable=var, style="Dark.TEntry", width=28).grid(
                row=row_idx, column=1, sticky="ew", padx=(10, 16), pady=5
            )
        section.columnconfigure(1, weight=1)

        context_section = tk.Frame(
            frame, bg=PANEL, highlightbackground=BORDER, highlightthickness=1
        )
        context_section.pack(fill="x", padx=24, pady=(0, 14))
        tk.Label(
            context_section,
            text="CONTEXTO DA CAMPANHA (armazenado, não usado como atalho neural)",
            bg=PANEL,
            fg="#a78bfa",
            font=("Segoe UI", 9, "bold"),
        ).grid(row=0, column=0, columnspan=2, sticky="w", padx=16, pady=(14, 10))

        for row_idx, (key, label) in enumerate(CONTEXT_FIELDS.items(), start=1):
            var = tk.StringVar()
            if existing_context.get(key) is not None:
                var.set(str(existing_context.get(key)))
            context_vars[key] = var
            tk.Label(context_section, text=label, bg=PANEL, fg=MUTED).grid(
                row=row_idx, column=0, sticky="w", padx=16, pady=5
            )
            ttk.Entry(
                context_section, textvariable=var, style="Dark.TEntry", width=28
            ).grid(row=row_idx, column=1, sticky="ew", padx=(10, 16), pady=5)
        context_section.columnconfigure(1, weight=1)

        status_var = tk.StringVar(value="")

        def save():
            try:
                result = save_campaign_metrics(
                    run_dir=run_dir,
                    output_root=Path(self.output.get()),
                    targets={k: v.get() for k, v in target_vars.items()},
                    context={k: v.get() for k, v in context_vars.items()},
                )
                training = result.get("training") or {}
                records = result.get("records", 0)
                trained = training.get("trained_model_count", 0)
                status_var.set(
                    f"Salvo • {records} criativo(s) no dataset • {trained} modelo(s) calibrado(s) treinado(s)."
                )
                messagebox.showinfo(
                    APP_NAME,
                    (
                        "Dados reais salvos.\n\n"
                        f"Dataset: {result.get('dataset_path')}\n"
                        f"Registros: {records}\n"
                        f"Modelos treinados: {trained}\n\n"
                        "Enquanto não houver amostras suficientes, nenhuma métrica "
                        "será tratada como previsão calibrada."
                    ),
                )
            except Exception as exc:
                messagebox.showerror(APP_NAME, f"Falha ao salvar calibração:\n\n{exc}")

        tk.Button(
            frame,
            text="SALVAR DADOS REAIS E ATUALIZAR CALIBRAÇÃO",
            command=save,
            bg=ACCENT,
            fg="white",
            activebackground="#6d28d9",
            activeforeground="white",
            relief="flat",
            padx=18,
            pady=12,
            font=("Segoe UI", 9, "bold"),
            cursor="hand2",
        ).pack(anchor="w", padx=24, pady=(0, 8))
        tk.Label(
            frame,
            textvariable=status_var,
            bg=BG,
            fg="#86efac",
            font=("Segoe UI", 9),
        ).pack(anchor="w", padx=24, pady=(0, 24))

    def _open_report(self):
        if self._last_report and self._last_report.exists():
            webbrowser.open(self._last_report.as_uri())


if __name__ == "__main__":
    App().mainloop()

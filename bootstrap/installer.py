from __future__ import annotations

import os
import shutil
import subprocess
import sys
import tempfile
import urllib.request
from pathlib import Path
import tkinter as tk
from tkinter import messagebox, ttk


APP_NAME = "TRIBE v2 Local Runner"
PY_VERSION = "3.11.9"
PY_INSTALLER_URL = (
    f"https://www.python.org/ftp/python/{PY_VERSION}/"
    f"python-{PY_VERSION}-amd64.exe"
)


def base_dir() -> Path:
    return Path(os.environ.get("LOCALAPPDATA", Path.home())) / "TRIBEv2LocalRunner"


def resource_dir() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parents[1]


def find_python() -> Path | None:
    target = base_dir() / "python" / "python.exe"
    if target.exists():
        return target
    for cmd in (["py", "-3.11", "-c", "import sys;print(sys.executable)"], ["python", "-c", "import sys;print(sys.executable)"]):
        try:
            out = subprocess.check_output(cmd, text=True, stderr=subprocess.DEVNULL).strip()
            p = Path(out)
            if p.exists() and sys.version_info[:2] >= (3, 11):
                return p
        except Exception:
            pass
    return None


def install_python(status):
    root = base_dir()
    root.mkdir(parents=True, exist_ok=True)
    target = root / "python"
    exe = Path(tempfile.gettempdir()) / f"python-{PY_VERSION}-amd64.exe"
    status("Downloading private Python runtime…")
    urllib.request.urlretrieve(PY_INSTALLER_URL, exe)
    status("Installing Python runtime…")
    subprocess.check_call(
        [
            str(exe),
            "/quiet",
            "InstallAllUsers=0",
            "PrependPath=0",
            "Include_launcher=0",
            "Include_test=0",
            "Include_pip=1",
            f"TargetDir={target}",
        ]
    )
    py = target / "python.exe"
    if not py.exists():
        raise RuntimeError("Python installer completed but python.exe was not found.")
    return py


def install_runtime(py: Path, status):
    root = base_dir()
    env = root / "env"
    src = resource_dir()
    runtime_src = src / "runtime"
    req_src = src / "requirements-runtime.txt"
    app_dst = root / "runtime"

    status("Creating isolated environment…")
    if not (env / "Scripts" / "python.exe").exists():
        subprocess.check_call([str(py), "-m", "venv", str(env)])
    vpy = env / "Scripts" / "python.exe"

    status("Updating installer tools…")
    subprocess.check_call([str(vpy), "-m", "pip", "install", "-U", "pip", "setuptools", "wheel"])

    status("Installing TRIBE v2 and local runtime (this can take a while)…")
    subprocess.check_call([str(vpy), "-m", "pip", "install", "-r", str(req_src)])

    status("Copying desktop app…")
    if app_dst.exists():
        shutil.rmtree(app_dst)
    shutil.copytree(runtime_src, app_dst)

    launcher = root / "Launch TRIBE v2.cmd"
    launcher.write_text(
        f'@echo off\r\nstart "" "{vpy}" "{app_dst / "app.py"}"\r\n',
        encoding="utf-8",
    )
    return vpy, app_dst / "app.py"


def main():
    win = tk.Tk()
    win.title(APP_NAME + " Setup")
    win.geometry("620x260")
    frame = ttk.Frame(win, padding=28)
    frame.pack(fill="both", expand=True)
    ttk.Label(frame, text=APP_NAME, font=("Segoe UI", 22, "bold")).pack(anchor="w")
    ttk.Label(
        frame,
        text="One-time setup. Python, PyTorch, TRIBE v2 runtime and support packages will be installed for this Windows user.",
        wraplength=555,
    ).pack(anchor="w", pady=(8, 20))
    status_var = tk.StringVar(value="Ready to install.")
    ttk.Label(frame, textvariable=status_var, wraplength=555).pack(anchor="w", pady=(0, 12))
    bar = ttk.Progressbar(frame, mode="indeterminate")
    bar.pack(fill="x")

    def set_status(text):
        status_var.set(text)
        win.update_idletasks()

    def go():
        button.configure(state="disabled")
        bar.start(12)
        try:
            py = find_python() or install_python(set_status)
            vpy, app = install_runtime(py, set_status)
            set_status("Setup complete. Launching…")
            subprocess.Popen([str(vpy), str(app)])
            messagebox.showinfo(APP_NAME, "Setup complete.")
            win.destroy()
        except Exception as exc:
            bar.stop()
            button.configure(state="normal")
            messagebox.showerror(APP_NAME, f"Setup failed:\n\n{exc}")
            set_status("Setup failed.")

    button = ttk.Button(frame, text="Install and launch", command=go)
    button.pack(anchor="w", pady=(16, 0))
    win.mainloop()


if __name__ == "__main__":
    main()

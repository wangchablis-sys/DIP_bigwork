"""SIED V3 极暗 RAW 图像增强桌面界面。

界面只调用同目录的 enhance_raw_v3_feature_ensemble.py。
它负责选择参数、显示进度和预览，不会改原始 RAW。
"""

from __future__ import annotations

import os
import queue
import re
import subprocess
import sys
import threading
from pathlib import Path
from tkinter import BooleanVar, IntVar, StringVar, TclError, Text, Tk, filedialog, messagebox, ttk

import cv2
import numpy as np
from PIL import Image, ImageOps, ImageTk


APP_DIR = Path(__file__).resolve().parent
V3_SCRIPT = APP_DIR / "enhance_raw_v3_feature_ensemble.py"
DEFAULT_MODEL_DIR = APP_DIR / "SIED_official"
DEFAULT_OUTPUT_DIR = APP_DIR / "output_v3"

PINK = "#e84d8a"
PINK_DARK = "#c93470"
PINK_LIGHT = "#ffd8e8"
PINK_PALE = "#fff4f8"
CARD = "#ffffff"
TEXT = "#4b3040"
MUTED = "#8d7180"


class PinkEnhancementApp:
    def __init__(self, root: Tk) -> None:
        self.root = root
        self.root.title("SIED V3 极暗 RAW 图像增强")
        self.root.geometry("1180x820")
        self.root.minsize(1040, 740)
        self.root.configure(bg=PINK_PALE)

        self.process: subprocess.Popen[str] | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.raw_photo: ImageTk.PhotoImage | None = None
        self.result_photo: ImageTk.PhotoImage | None = None

        self.input_var = StringVar()
        self.output_var = StringVar(value=str(DEFAULT_OUTPUT_DIR))
        self.camera_var = StringVar(value="canon")
        self.lux_var = StringVar(value="0.0001")
        self.ckpt_var = StringVar()
        self.model_var = StringVar(value=str(DEFAULT_MODEL_DIR))
        self.steps_var = IntVar(value=20)
        self.samples_var = IntVar(value=3)
        self.tile_var = IntVar(value=512)
        self.overlap_var = IntVar(value=64)
        self.limit_var = IntVar(value=0)
        self.overwrite_var = BooleanVar(value=False)
        self.status_var = StringVar(value="准备就绪")
        self.progress_text_var = StringVar(value="尚未开始")

        self._configure_styles()
        self._build_ui()
        self._poll_events()
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

    def _configure_styles(self) -> None:
        style = ttk.Style(self.root)
        style.theme_use("clam")
        style.configure("App.TFrame", background=PINK_PALE)
        style.configure("Card.TFrame", background=CARD)
        style.configure("Header.TLabel", background=PINK, foreground="white", font=("Microsoft YaHei UI", 22, "bold"))
        style.configure("SubHeader.TLabel", background=PINK, foreground="#ffeaf3", font=("Microsoft YaHei UI", 10))
        style.configure("CardTitle.TLabel", background=CARD, foreground=TEXT, font=("Microsoft YaHei UI", 12, "bold"))
        style.configure("Body.TLabel", background=CARD, foreground=TEXT, font=("Microsoft YaHei UI", 10))
        style.configure("Muted.TLabel", background=CARD, foreground=MUTED, font=("Microsoft YaHei UI", 9))
        style.configure("Status.TLabel", background=PINK_LIGHT, foreground=PINK_DARK, font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 6))
        style.configure("Pink.TButton", background=PINK, foreground="white", borderwidth=0, font=("Microsoft YaHei UI", 10, "bold"), padding=(12, 8))
        style.map("Pink.TButton", background=[("active", PINK_DARK), ("disabled", "#d9aabb")])
        style.configure("Soft.TButton", background=PINK_LIGHT, foreground=PINK_DARK, borderwidth=0, font=("Microsoft YaHei UI", 9), padding=(10, 7))
        style.map("Soft.TButton", background=[("active", "#ffc3db")])
        style.configure("Danger.TButton", background="#f3ccd9", foreground="#a52d58", borderwidth=0, padding=(10, 8))
        style.configure("Pink.Horizontal.TProgressbar", troughcolor="#f7d7e4", background=PINK, lightcolor=PINK, darkcolor=PINK)
        style.configure("Pink.TCheckbutton", background=CARD, foreground=TEXT, font=("Microsoft YaHei UI", 9))

    def _build_ui(self) -> None:
        header = ttk.Frame(self.root, style="App.TFrame")
        header.pack(fill="x")
        banner = ttk.Frame(header, style="App.TFrame", height=92)
        banner.pack(fill="x")
        banner.pack_propagate(False)
        ttk.Label(banner, text="  SIED V3 极暗 RAW 图像增强", style="Header.TLabel", anchor="w").place(x=0, y=0, relwidth=1, height=58)
        ttk.Label(
            banner,
            text="  FP32 · 默认 20 步 × 3 次特征融合 · PNG 写入 output_v3 · 不改原图",
            style="SubHeader.TLabel",
            anchor="w",
        ).place(x=0, y=56, relwidth=1, height=36)

        content = ttk.Frame(self.root, style="App.TFrame", padding=14)
        content.pack(fill="both", expand=True)
        content.columnconfigure(0, weight=0, minsize=460)
        content.columnconfigure(1, weight=1)
        content.rowconfigure(0, weight=1)

        left = ttk.Frame(content, style="App.TFrame")
        left.grid(row=0, column=0, sticky="nsew", padx=(0, 12))
        right = ttk.Frame(content, style="App.TFrame")
        right.grid(row=0, column=1, sticky="nsew")
        self._build_controls(left)
        self._build_preview(right)

    def _card(self, parent: ttk.Frame, title: str) -> ttk.Frame:
        outer = ttk.Frame(parent, style="Card.TFrame", padding=14)
        outer.pack(fill="x", pady=(0, 10))
        ttk.Label(outer, text=title, style="CardTitle.TLabel").pack(anchor="w", pady=(0, 10))
        return outer

    def _path_row(self, parent: ttk.Frame, variable: StringVar, command, button_text: str) -> None:
        row = ttk.Frame(parent, style="Card.TFrame")
        row.pack(fill="x", pady=4)
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True, padx=(0, 7))
        ttk.Button(row, text=button_text, style="Soft.TButton", command=command).pack(side="right")

    def _build_controls(self, parent: ttk.Frame) -> None:
        paths = self._card(parent, "① 选择图片")
        ttk.Label(paths, text="RAW 文件或包含 RAW 的文件夹", style="Body.TLabel").pack(anchor="w")
        input_row = ttk.Frame(paths, style="Card.TFrame")
        input_row.pack(fill="x", pady=4)
        ttk.Entry(input_row, textvariable=self.input_var).pack(side="left", fill="x", expand=True, padx=(0, 7))
        ttk.Button(input_row, text="文件", style="Soft.TButton", command=self._choose_file).pack(side="right", padx=(0, 6))
        ttk.Button(input_row, text="文件夹", style="Soft.TButton", command=self._choose_folder).pack(side="right")
        ttk.Label(paths, text="增强结果输出文件夹", style="Body.TLabel").pack(anchor="w", pady=(5, 0))
        self._path_row(paths, self.output_var, self._choose_output, "选择")

        params = self._card(parent, "② V3 参数")
        grid = ttk.Frame(params, style="Card.TFrame")
        grid.pack(fill="x")
        for col in range(4):
            grid.columnconfigure(col, weight=1)

        self._grid_label(grid, "相机", 0, 0)
        ttk.Combobox(grid, textvariable=self.camera_var, values=("canon", "sony"), state="readonly", width=8).grid(
            row=1, column=0, sticky="ew", padx=(0, 6), pady=(2, 8)
        )
        self._grid_label(grid, "照度", 0, 1)
        ttk.Combobox(grid, textvariable=self.lux_var, values=("0.0001", "0.001", "0.01"), state="readonly", width=8).grid(
            row=1, column=1, sticky="ew", padx=6, pady=(2, 8)
        )
        self._grid_label(grid, "采样步数", 0, 2)
        ttk.Spinbox(grid, from_=1, to=1000, textvariable=self.steps_var, width=8).grid(
            row=1, column=2, sticky="ew", padx=6, pady=(2, 8)
        )
        self._grid_label(grid, "融合次数", 0, 3)
        ttk.Spinbox(grid, from_=1, to=16, textvariable=self.samples_var, width=8).grid(
            row=1, column=3, sticky="ew", padx=(6, 0), pady=(2, 8)
        )

        self._grid_label(grid, "处理张数（0=全部）", 2, 0)
        ttk.Spinbox(grid, from_=0, to=10000, textvariable=self.limit_var, width=8).grid(
            row=3, column=0, sticky="ew", padx=(0, 6), pady=(2, 8)
        )
        self._grid_label(grid, "分块尺寸", 2, 1)
        ttk.Combobox(grid, textvariable=self.tile_var, values=(256, 512, 768, 1024), state="readonly", width=8).grid(
            row=3, column=1, sticky="ew", padx=6, pady=(2, 8)
        )
        self._grid_label(grid, "重叠像素", 2, 2)
        ttk.Combobox(grid, textvariable=self.overlap_var, values=(32, 64, 96, 128), state="readonly", width=8).grid(
            row=3, column=2, sticky="ew", padx=6, pady=(2, 8)
        )
        ttk.Checkbutton(grid, text="覆盖已有结果", variable=self.overwrite_var, style="Pink.TCheckbutton").grid(
            row=3, column=3, sticky="w", padx=(6, 0), pady=(2, 8)
        )
        ttk.Label(params, text="分块尺寸和重叠像素只在显存不足、整张放不下时使用。", style="Muted.TLabel", wraplength=410).pack(anchor="w")

        ttk.Label(params, text="权重文件（必填 .pt）", style="Body.TLabel").pack(anchor="w", pady=(8, 0))
        self._path_row(params, self.ckpt_var, self._choose_ckpt, "选择")
        ttk.Label(params, text="模型代码目录", style="Body.TLabel").pack(anchor="w", pady=(4, 0))
        self._path_row(params, self.model_var, self._choose_model, "选择")

        action = self._card(parent, "③ 开始处理")
        buttons = ttk.Frame(action, style="Card.TFrame")
        buttons.pack(fill="x")
        self.start_button = ttk.Button(buttons, text="开始增强", style="Pink.TButton", command=self._start)
        self.start_button.pack(side="left", fill="x", expand=True, padx=(0, 7))
        self.stop_button = ttk.Button(buttons, text="停止", style="Danger.TButton", command=self._stop, state="disabled")
        self.stop_button.pack(side="left", padx=(0, 7))
        ttk.Button(buttons, text="打开输出", style="Soft.TButton", command=self._open_output).pack(side="left")
        self.progress = ttk.Progressbar(action, style="Pink.Horizontal.TProgressbar", mode="determinate", maximum=100)
        self.progress.pack(fill="x", pady=(12, 5))
        info = ttk.Frame(action, style="Card.TFrame")
        info.pack(fill="x")
        ttk.Label(info, textvariable=self.progress_text_var, style="Muted.TLabel").pack(side="left")
        ttk.Label(info, textvariable=self.status_var, style="Status.TLabel").pack(side="right")

    @staticmethod
    def _grid_label(parent: ttk.Frame, text: str, row: int, column: int) -> None:
        ttk.Label(parent, text=text, style="Body.TLabel").grid(row=row, column=column, sticky="w", padx=(0 if column == 0 else 6, 0))

    def _build_preview(self, parent: ttk.Frame) -> None:
        preview = ttk.Frame(parent, style="Card.TFrame", padding=14)
        preview.pack(fill="both", expand=True, pady=(0, 10))
        top = ttk.Frame(preview, style="Card.TFrame")
        top.pack(fill="x", pady=(0, 8))
        ttk.Label(top, text="图像预览", style="CardTitle.TLabel").pack(side="left")
        ttk.Button(top, text="刷新预览", style="Soft.TButton", command=self._refresh_preview).pack(side="right")

        images = ttk.Frame(preview, style="Card.TFrame")
        images.pack(fill="both", expand=True)
        images.columnconfigure(0, weight=1)
        images.columnconfigure(1, weight=1)
        images.rowconfigure(1, weight=1)
        ttk.Label(images, text="RAW 快速预览", style="Body.TLabel").grid(row=0, column=0, pady=(0, 5))
        ttk.Label(images, text="增强结果", style="Body.TLabel").grid(row=0, column=1, pady=(0, 5))
        self.raw_label = ttk.Label(images, text="选择 RAW 后显示", style="Muted.TLabel", anchor="center")
        self.raw_label.grid(row=1, column=0, sticky="nsew", padx=(0, 5))
        self.result_label = ttk.Label(images, text="处理完成后显示", style="Muted.TLabel", anchor="center")
        self.result_label.grid(row=1, column=1, sticky="nsew", padx=(5, 0))

        log_card = ttk.Frame(parent, style="Card.TFrame", padding=14)
        log_card.pack(fill="both")
        ttk.Label(log_card, text="运行日志", style="CardTitle.TLabel").pack(anchor="w", pady=(0, 7))
        log_frame = ttk.Frame(log_card, style="Card.TFrame")
        log_frame.pack(fill="both", expand=True)
        self.log = Text(
            log_frame, height=8, wrap="word", bg="#3d2734", fg="#ffeaf3",
            insertbackground="white", relief="flat", padx=10, pady=8,
            font=("Consolas", 9), state="disabled",
        )
        scroll = ttk.Scrollbar(log_frame, orient="vertical", command=self.log.yview)
        self.log.configure(yscrollcommand=scroll.set)
        self.log.pack(side="left", fill="both", expand=True)
        scroll.pack(side="right", fill="y")

    def _choose_file(self) -> None:
        path = filedialog.askopenfilename(title="选择一张 RAW", filetypes=[("RAW 图像", "*.raw"), ("所有文件", "*.*")])
        if path:
            self.input_var.set(path)
            self._refresh_preview()

    def _choose_folder(self) -> None:
        path = filedialog.askdirectory(title="选择包含 RAW 的文件夹")
        if path:
            self.input_var.set(path)
            self._refresh_preview()

    def _choose_output(self) -> None:
        path = filedialog.askdirectory(title="选择输出文件夹")
        if path:
            self.output_var.set(path)

    def _choose_ckpt(self) -> None:
        path = filedialog.askopenfilename(
            title="选择权重文件",
            filetypes=[("PyTorch 权重", "*.pt"), ("所有文件", "*.*")],
        )
        if path:
            self.ckpt_var.set(path)

    def _choose_model(self) -> None:
        path = filedialog.askdirectory(title="选择 SIED 官方代码目录")
        if path:
            self.model_var.set(path)

    def _first_raw(self) -> Path | None:
        text = self.input_var.get().strip().strip('"')
        if not text:
            return None
        path = Path(text)
        if path.is_file() and path.suffix.lower() == ".raw":
            return path
        if path.is_dir():
            raws = sorted(path.glob("*.raw"), key=lambda item: (0, int(item.stem)) if item.stem.isdigit() else (1, item.name.lower()))
            return raws[0] if raws else None
        return None

    def _refresh_preview(self) -> None:
        raw_path = self._first_raw()
        if raw_path is None:
            return
        try:
            raw = np.fromfile(raw_path, np.uint16)
            if raw.size != 2160 * 3840:
                raise ValueError("RAW 尺寸不是 3840×2160")
            raw = raw.reshape(2160, 3840).astype(np.float32)
            raw = np.clip((raw - 512.0) / (16383.0 - 512.0), 0, 1)
            high = float(np.percentile(raw[::4, ::4], 99.7))
            preview = np.clip(raw / max(high, 1e-5) * 0.82, 0, 1)
            preview16 = np.round(preview * 65535).astype(np.uint16)
            bgr = cv2.cvtColor(preview16, cv2.COLOR_BAYER_RG2BGR)
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            raw_image = Image.fromarray((rgb / 257).astype(np.uint8))
            self._show_image(raw_image, self.raw_label, "raw")

            output = Path(self.output_var.get().strip().strip('"')) / f"{raw_path.stem}_enhanced.png"
            if output.is_file():
                with Image.open(output) as image:
                    self._show_image(image.convert("RGB"), self.result_label, "result")
            else:
                self.result_label.configure(image="", text="尚未生成对应结果")
                self.result_photo = None
        except Exception as exc:
            self._append_log(f"预览失败：{exc}\n")

    def _show_image(self, image: Image.Image, label: ttk.Label, kind: str) -> None:
        box = (470, 320)
        fitted = ImageOps.contain(image, box, Image.Resampling.LANCZOS)
        canvas = Image.new("RGB", box, "#f8eaf0")
        canvas.paste(fitted, ((box[0] - fitted.width) // 2, (box[1] - fitted.height) // 2))
        photo = ImageTk.PhotoImage(canvas)
        label.configure(image=photo, text="")
        if kind == "raw":
            self.raw_photo = photo
        else:
            self.result_photo = photo

    def _read_int(self, variable: IntVar, name: str) -> int:
        try:
            return int(variable.get())
        except (TclError, TypeError, ValueError):
            raise ValueError(f"{name} 需要是整数。") from None

    def _build_command(self) -> list[str]:
        if not V3_SCRIPT.is_file():
            raise FileNotFoundError(f"找不到 V3 脚本：{V3_SCRIPT}")

        input_path = Path(self.input_var.get().strip().strip('"'))
        output_path = Path(self.output_var.get().strip().strip('"'))
        if not input_path.exists():
            raise FileNotFoundError("请选择有效的 RAW 文件或文件夹。")
        if not str(output_path):
            raise ValueError("请选择输出文件夹。")

        steps = self._read_int(self.steps_var, "采样步数")
        samples = self._read_int(self.samples_var, "融合次数")
        limit = self._read_int(self.limit_var, "处理张数")
        tile = self._read_int(self.tile_var, "分块尺寸")
        overlap = self._read_int(self.overlap_var, "重叠像素")
        if not 1 <= steps <= 1000:
            raise ValueError("采样步数必须在 1 到 1000 之间。")
        if samples < 1:
            raise ValueError("融合次数至少为 1。")
        if limit < 0:
            raise ValueError("处理张数不能小于 0。")
        if tile % 256 != 0:
            raise ValueError("分块尺寸必须是 256 的倍数。")
        if overlap < 0:
            raise ValueError("重叠像素不能小于 0。")

        ckpt_text = self.ckpt_var.get().strip().strip('"')
        if not ckpt_text:
            raise ValueError("请选择权重文件。")
        ckpt = Path(ckpt_text)
        if not ckpt.is_file():
            raise FileNotFoundError(f"找不到权重：{ckpt}\n请输入或选择一个 .pt 文件。")
        model_dir = Path(self.model_var.get().strip().strip('"'))
        if not (model_dir / "models").is_dir():
            raise FileNotFoundError(f"找不到模型代码：{model_dir}")

        output_path.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable, "-u", str(V3_SCRIPT), str(input_path),
            "--output", str(output_path),
            "--ckpt", str(ckpt),
            "--model-dir", str(model_dir),
            "--camera", self.camera_var.get(),
            "--lux", self.lux_var.get(),
            "--steps", str(steps),
            "--samples", str(samples),
            "--tile", str(tile),
            "--overlap", str(overlap),
            "--limit", str(limit),
        ]
        if self.overwrite_var.get():
            command.append("--overwrite")
        return command

    def _start(self) -> None:
        if self.process is not None:
            return
        try:
            command = self._build_command()
        except Exception as exc:
            messagebox.showerror("无法开始", str(exc))
            return
        self.log.configure(state="normal")
        self.log.delete("1.0", "end")
        self.log.configure(state="disabled")
        self.progress["value"] = 0
        self.progress_text_var.set("正在启动模型……")
        self.status_var.set("运行中")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._append_log(subprocess.list2cmdline(command) + "\n\n")
        self._append_log("首次加载约 2.6GB 权重需要一些时间。\n\n")
        threading.Thread(target=self._run_worker, args=(command,), daemon=True).start()

    def _run_worker(self, command: list[str]) -> None:
        flags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0
        try:
            self.process = subprocess.Popen(
                command, cwd=APP_DIR, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, encoding="utf-8", errors="replace", bufsize=1,
                creationflags=flags,
            )
            assert self.process.stdout is not None
            for line in self.process.stdout:
                self.events.put(("line", line))
            code = self.process.wait()
            self.events.put(("done", code))
        except Exception as exc:
            self.events.put(("error", str(exc)))

    def _poll_events(self) -> None:
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "line":
                    line = str(value)
                    self._append_log(line)
                    match = re.search(r"\[(\d+)/(\d+)]", line)
                    if match:
                        done, total = map(int, match.groups())
                        self.progress["value"] = done / max(total, 1) * 100
                        self.progress_text_var.set(f"已处理 {done}/{total} 张")
                elif kind == "done":
                    self._finish(int(value) == 0, f"进程结束，返回码 {value}")
                elif kind == "error":
                    self._append_log(f"\n启动失败：{value}\n")
                    self._finish(False, str(value))
        except queue.Empty:
            pass
        self.root.after(120, self._poll_events)

    def _finish(self, success: bool, detail: str) -> None:
        self.process = None
        self.start_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        self.status_var.set("处理完成" if success else "处理失败")
        if success:
            self.progress["value"] = 100
            self.progress_text_var.set("全部任务已结束")
            self._refresh_preview()
        else:
            self.progress_text_var.set(detail)

    def _stop(self) -> None:
        if self.process is None:
            return
        if messagebox.askyesno("停止处理", "确定停止当前增强任务吗？已生成的 PNG 会保留。"):
            self.process.terminate()
            self.status_var.set("正在停止")
            self._append_log("\n用户请求停止任务……\n")

    def _open_output(self) -> None:
        path = Path(self.output_var.get().strip().strip('"'))
        path.mkdir(parents=True, exist_ok=True)
        if os.name == "nt":
            os.startfile(path)  # type: ignore[attr-defined]

    def _append_log(self, text: str) -> None:
        self.log.configure(state="normal")
        self.log.insert("end", text)
        self.log.see("end")
        self.log.configure(state="disabled")

    def _on_close(self) -> None:
        if self.process is not None:
            if not messagebox.askyesno("退出", "增强任务仍在运行，确定停止并退出吗？"):
                return
            self.process.terminate()
        self.root.destroy()


def main() -> None:
    root = Tk()
    PinkEnhancementApp(root)
    root.mainloop()


if __name__ == "__main__":
    main()

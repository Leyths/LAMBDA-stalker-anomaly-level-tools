#!/usr/bin/env python3
"""
L.A.M.B.D.A. Build System GUI

Graphical frontend for the LAMBDA compiler (build_all_spawn.py).
"""

import json
import os
import platform
import queue
import re
import subprocess
import sys
import threading
import time
import traceback
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path

# Path resolution: frozen EXE vs dev mode
if getattr(sys, 'frozen', False):
    PROJECT_ROOT = Path(sys.executable).parent.resolve()
else:
    PROJECT_ROOT = Path(__file__).parent.resolve()

COMPILER_DIR = PROJECT_ROOT / "compiler"
SETTINGS_FILE = PROJECT_ROOT / ".lambda_settings.json"

# Bundled assets: in frozen EXE they're extracted to sys._MEIPASS, not next to the EXE
ASSETS_DIR = Path(sys._MEIPASS) if getattr(sys, 'frozen', False) else PROJECT_ROOT
LOGO_PATH = ASSETS_DIR / "docs" / "images" / "lambda_logo.png"
ICON_PATH = ASSETS_DIR / "docs" / "images" / "icon.png"

ANSI_RE = re.compile(r'\033\[[0-9;]*m')

MOD_FLAVOURS = ["anomaly", "cultured", "gamma"]

if platform.system() == "Windows":
    MONO_FONT = ("Consolas", 10)
elif platform.system() == "Darwin":
    MONO_FONT = ("Menlo", 11)
else:
    MONO_FONT = ("Monospace", 10)

DEFAULTS = {
    "levels_dir": str(PROJECT_ROOT / "levels"),
    "output_dir": str(PROJECT_ROOT / "gamedata"),
    "levels_override_dir": "",
    "base_mod": "anomaly",
    "window_geometry": "900x650",
    "vis_spawn_path": "",
    "vis_level": "",
}


class QueueStream:
    """File-like object that writes to a queue for GUI output capture."""
    def __init__(self, q):
        self.queue = q

    def write(self, text):
        if text:
            self.queue.put(text)

    def flush(self):
        pass


class LambdaGUI:
    def __init__(self, root: tk.Tk, build_overrides=None):
        self.root = root
        self.root.title("L.A.M.B.D.A. Build System")
        self.root.minsize(600, 400)

        # Window icon
        if ICON_PATH.exists():
            icon = tk.PhotoImage(file=str(ICON_PATH))
            self.root.iconphoto(True, icon)
            self._icon_ref = icon  # prevent garbage collection

        # macOS aqua theme ignores button padding/sizing - use clam instead
        style = ttk.Style()
        if style.theme_use() == "aqua":
            style.theme_use("clam")

        self.build_thread = None
        self.output_queue = queue.Queue()
        self.build_cancelled = False
        self.build_start_time = None
        self.elapsed_after_id = None

        # Settings variables
        self.levels_dir_var = tk.StringVar(value=DEFAULTS["levels_dir"])
        self.output_dir_var = tk.StringVar(value=DEFAULTS["output_dir"])
        self.levels_override_dir_var = tk.StringVar(value=DEFAULTS["levels_override_dir"])
        self.base_mod_var = tk.StringVar(value=DEFAULTS["base_mod"])

        # Visualiser settings variables
        self.vis_spawn_var = tk.StringVar()
        self.vis_level_var = tk.StringVar()
        self._vis_levels = []

        self._load_settings()

        # Apply CLI overrides (from --build mode)
        if build_overrides:
            if "levels_dir" in build_overrides:
                self.levels_dir_var.set(build_overrides["levels_dir"])
            if "output_dir" in build_overrides:
                self.output_dir_var.set(build_overrides["output_dir"])
            if "base_mod" in build_overrides:
                self.base_mod_var.set(build_overrides["base_mod"])
            if "levels_override_dir" in build_overrides:
                self.levels_override_dir_var.set(build_overrides["levels_override_dir"])

        self._create_widgets()

        self.root.protocol("WM_DELETE_WINDOW", self._on_close)

        # Auto-start build if launched with --build
        if build_overrides is not None:
            deploy_only = build_overrides.get("deploy_only", False)
            self.root.after(100, lambda: self._start_build(deploy_only=deploy_only))

    def _create_widgets(self):
        # Main container
        main = ttk.Frame(self.root, padding=10)
        main.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main.columnconfigure(0, weight=1)
        main.rowconfigure(0, weight=1)

        # --- Notebook (tabs) ---
        self.notebook = ttk.Notebook(main)
        self.notebook.grid(row=0, column=0, sticky="nsew")

        # --- Build tab ---
        build_tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(build_tab, text="Build")
        build_tab.columnconfigure(0, weight=1)
        build_tab.rowconfigure(1, weight=1)

        self._create_build_tab(build_tab)

        # --- Visualiser tab ---
        vis_tab = ttk.Frame(self.notebook, padding=4)
        self.notebook.add(vis_tab, text="Visualiser")
        vis_tab.columnconfigure(0, weight=1)

        self._create_visualiser_tab(vis_tab)

        # Refresh level list when switching to Visualiser tab
        self.notebook.bind("<<NotebookTabChanged>>", self._on_tab_changed)

    def _create_build_tab(self, parent):
        # --- Settings frame ---
        settings = ttk.Frame(parent, padding=(0, 0, 0, 8))
        settings.grid(row=0, column=0, sticky="ew")

        # Levels directory (row 0)
        ttk.Label(settings, text="Levels Directory:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.levels_entry = ttk.Entry(settings, textvariable=self.levels_dir_var, width=50)
        self.levels_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        self.levels_browse_btn = ttk.Button(settings, text="...", width=3,
                                            command=lambda: self._browse_dir(self.levels_dir_var))
        self.levels_browse_btn.grid(row=0, column=2)

        # Runtime Levels Directory (row 1) — only shown if it has a value
        self._build_override_label = ttk.Label(settings, text="Runtime Levels Directory:")
        self.override_entry = ttk.Entry(settings, textvariable=self.levels_override_dir_var, width=50)
        self.override_browse_btn = ttk.Button(settings, text="...", width=3,
                                              command=lambda: self._browse_dir(self.levels_override_dir_var))
        if self.levels_override_dir_var.get():
            self._build_override_label.grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
            self.override_entry.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=(4, 0))
            self.override_browse_btn.grid(row=1, column=2, pady=(4, 0))

        # Output directory (row 2)
        ttk.Label(settings, text="Output Directory:").grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.output_entry = ttk.Entry(settings, textvariable=self.output_dir_var, width=50)
        self.output_entry.grid(row=2, column=1, sticky="ew", padx=(0, 4), pady=(4, 0))
        self.output_browse_btn = ttk.Button(settings, text="...", width=3,
                                            command=lambda: self._browse_dir(self.output_dir_var))
        self.output_browse_btn.grid(row=2, column=2, pady=(4, 0))

        # Base mod (row 3)
        ttk.Label(settings, text="Base Mod:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.mod_combo = ttk.Combobox(settings, textvariable=self.base_mod_var, values=MOD_FLAVOURS,
                                      state="readonly", width=15)
        self.mod_combo.grid(row=3, column=1, sticky="w", pady=(4, 0))

        settings.columnconfigure(1, weight=1)

        # --- Output frame ---
        output_frame = ttk.Frame(parent, padding=0)
        output_frame.grid(row=1, column=0, sticky="nsew", pady=(0, 8))

        self.output_text = tk.Text(output_frame, font=MONO_FONT, wrap="char", state="disabled",
                                   bg="#1e1e1e", fg="#d4d4d4", insertbackground="#d4d4d4",
                                   selectbackground="#264f78", selectforeground="#d4d4d4",
                                   padx=8)
        self.output_text.grid(row=0, column=0, sticky="nsew")

        scrollbar = ttk.Scrollbar(output_frame, orient="vertical", command=self.output_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.output_text.configure(yscrollcommand=scrollbar.set)

        output_frame.columnconfigure(0, weight=1)
        output_frame.rowconfigure(0, weight=1)

        # Text tags for colorization
        self.output_text.tag_configure("warning", foreground="#dcdcaa")
        self.output_text.tag_configure("error", foreground="#f44747")
        self.output_text.tag_configure("step", foreground="#569cd6", font=(MONO_FONT[0], MONO_FONT[1], "bold"))
        self.output_text.tag_configure("complete", foreground="#6a9955", font=(MONO_FONT[0], MONO_FONT[1], "bold"))
        self.output_text.tag_configure("override", foreground="#569cd6")
        self.output_text.tag_configure("base_source", foreground="#6a9955")

        # --- Bottom bar ---
        bottom = ttk.Frame(parent)
        bottom.grid(row=2, column=0, sticky="ew")

        # Logo (bottom-left, small)
        if LOGO_PATH.exists():
            raw = tk.PhotoImage(file=str(LOGO_PATH))
            scale = max(1, raw.width() // 267)
            self._logo_image = raw.subsample(scale, scale)
            logo_label = ttk.Label(bottom, image=self._logo_image)
            logo_label.pack(side="left", anchor="s")

        # Buttons (right side, stacked)
        btn_frame = ttk.Frame(bottom)
        btn_frame.pack(side="right")

        style = ttk.Style()
        style.configure("FullBuild.TButton", font=(MONO_FONT[0], 13, "bold"), padding=(20, 12))
        style.configure("Cancel.TButton", font=(MONO_FONT[0], 13, "bold"), padding=(20, 12))
        style.configure("Deploy.TButton", font=(MONO_FONT[0], 10), padding=(20, 4))

        self.full_build_btn = ttk.Button(btn_frame, text="Full Build", command=self._on_full_build,
                                         style="FullBuild.TButton", cursor="hand2")
        self.full_build_btn.pack(pady=(0, 0), ipady=10, fill="x")

        self.elapsed_label = ttk.Label(btn_frame, text="", font=(MONO_FONT[0], 9),
                                        anchor="center")
        self.elapsed_label.pack(fill="x", pady=(0, 4))

        self.deploy_btn = ttk.Button(btn_frame, text="Deploy Scripts Only", command=self._on_deploy,
                                     style="Deploy.TButton", cursor="hand2")
        self.deploy_btn.pack(fill="x")

    def _create_visualiser_tab(self, parent):
        # --- Visualiser settings ---
        vis_settings = ttk.Frame(parent, padding=(0, 8, 0, 8))
        vis_settings.grid(row=0, column=0, sticky="ew")
        vis_settings.columnconfigure(1, weight=1)

        # Path to all.spawn (row 0)
        ttk.Label(vis_settings, text="Path to all.spawn:").grid(row=0, column=0, sticky="w", padx=(0, 8))
        self.vis_spawn_entry = ttk.Entry(vis_settings, textvariable=self.vis_spawn_var, width=50)
        self.vis_spawn_entry.grid(row=0, column=1, sticky="ew", padx=(0, 4))
        ttk.Button(vis_settings, text="...", width=3,
                   command=self._browse_spawn_file).grid(row=0, column=2)

        # Levels Directory (row 1) — shared with Build tab
        ttk.Label(vis_settings, text="Levels Directory:").grid(row=1, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.vis_levels_entry = ttk.Entry(vis_settings, textvariable=self.levels_dir_var, width=50)
        self.vis_levels_entry.grid(row=1, column=1, sticky="ew", padx=(0, 4), pady=(4, 0))
        ttk.Button(vis_settings, text="...", width=3,
                   command=lambda: self._browse_dir(self.levels_dir_var)).grid(row=1, column=2, pady=(4, 0))

        # Runtime Levels Directory (row 2) — only shown if it has a value
        self._vis_override_label = ttk.Label(vis_settings, text="Runtime Levels Directory:")
        self._vis_override_entry = ttk.Entry(vis_settings, textvariable=self.levels_override_dir_var, width=50)
        self._vis_override_browse = ttk.Button(vis_settings, text="...", width=3,
                                               command=lambda: self._browse_dir(self.levels_override_dir_var))
        if self.levels_override_dir_var.get():
            self._vis_override_label.grid(row=2, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
            self._vis_override_entry.grid(row=2, column=1, sticky="ew", padx=(0, 4), pady=(4, 0))
            self._vis_override_browse.grid(row=2, column=2, pady=(4, 0))

        # Level dropdown (row 3)
        ttk.Label(vis_settings, text="Select level:").grid(row=3, column=0, sticky="w", padx=(0, 8), pady=(4, 0))
        self.vis_level_combo = ttk.Combobox(vis_settings, textvariable=self.vis_level_var,
                                            state="readonly", width=30)
        self.vis_level_combo.grid(row=3, column=1, sticky="ew", pady=(4, 0))

        # View Level button (row 4)
        style = ttk.Style()
        style.configure("ViewLevel.TButton", font=(MONO_FONT[0], 13, "bold"), padding=(20, 12))
        self.view_level_btn = ttk.Button(vis_settings, text="View Level", command=self._on_view_level,
                                         style="ViewLevel.TButton", cursor="hand2")
        self.view_level_btn.grid(row=4, column=1, sticky="w", pady=(16, 0), ipady=6)

        # Loading status label (row 5)
        self.vis_status_label = ttk.Label(vis_settings, text="", font=(MONO_FONT[0], 9))
        self.vis_status_label.grid(row=5, column=1, sticky="w", pady=(4, 0))

    # --- Visualiser methods ---

    def _browse_spawn_file(self):
        current = self.vis_spawn_var.get()
        initial_dir = str(Path(current).parent) if current and Path(current).parent.is_dir() else str(PROJECT_ROOT)
        path = filedialog.askopenfilename(
            initialdir=initial_dir,
            title="Select all.spawn",
            filetypes=[("Spawn files", "*.spawn"), ("All files", "*.*")],
        )
        if path:
            self.vis_spawn_var.set(str(Path(path).resolve()))

    def _on_tab_changed(self, event):
        selected = self.notebook.index(self.notebook.select())
        if selected == 1:
            self._refresh_vis_levels()

    def _resolve_level_ai_path(self, level_name: str):
        """Find level.ai for a level, checking override dir first."""
        override_dir = self.levels_override_dir_var.get()
        levels_dir = self._resolve_path(self.levels_dir_var.get())

        if override_dir:
            override_path = Path(override_dir) / level_name / "level.ai"
            if override_path.exists():
                return str(override_path)

        base_path = levels_dir / level_name / "level.ai"
        if base_path.exists():
            return str(base_path)

        return None

    def _refresh_vis_levels(self):
        spawn_path = self.vis_spawn_var.get()
        if not spawn_path or not Path(spawn_path).exists():
            self._vis_levels = []
            self.vis_level_combo["values"] = []
            return

        try:
            # Ensure compiler/ is on sys.path (dev mode; in frozen EXE these are bundled)
            compiler_dir = str(COMPILER_DIR)
            if compiler_dir not in sys.path:
                sys.path.insert(0, compiler_dir)
            from parsers import GameGraphParser

            gg = GameGraphParser.from_all_spawn(Path(spawn_path))
            levels = gg.get_levels()

            available = []
            for level_id in sorted(levels.keys()):
                level = levels[level_id]
                ai_path = self._resolve_level_ai_path(level.name)
                available.append({
                    'name': level.name,
                    'level_id': level_id,
                    'ai_path': ai_path,
                })

            self._vis_levels = available
            level_names = [l['name'] for l in available]
            self.vis_level_combo["values"] = level_names

            # Restore previous selection if still valid
            current = self.vis_level_var.get()
            if current not in level_names:
                self.vis_level_var.set("")
        except Exception as e:
            import traceback
            traceback.print_exc()
            self.vis_level_combo["values"] = [f"Error: {e}"]

    def _on_view_level(self):
        level_name = self.vis_level_var.get()
        if not level_name:
            messagebox.showwarning("No level", "Please select a level.")
            return

        all_spawn = self.vis_spawn_var.get()
        if not all_spawn or not Path(all_spawn).exists():
            messagebox.showerror("Not found", f"all.spawn not found:\n{all_spawn}")
            return

        level_info = next((l for l in self._vis_levels if l['name'] == level_name), None)
        if level_info is None:
            messagebox.showerror("Error", f"Level '{level_name}' not found.")
            return

        ai_path = level_info['ai_path']

        if ai_path is None or not Path(ai_path).exists():
            # Build a helpful message showing which directories were checked
            levels_dir = self._resolve_path(self.levels_dir_var.get())
            override_dir = self.levels_override_dir_var.get()
            checked = [str(levels_dir / level_name / "level.ai")]
            if override_dir:
                checked.insert(0, str(Path(override_dir) / level_name / "level.ai"))
            messagebox.showerror("Not found",
                                 f"level.ai not found for '{level_name}'.\n\n"
                                 f"Checked:\n" + "\n".join(f"  {p}" for p in checked))
            return

        # GUID validation: check level.ai matches the all.spawn cross table
        try:
            compiler_dir = str(COMPILER_DIR)
            if compiler_dir not in sys.path:
                sys.path.insert(0, compiler_dir)
            from parsers import LevelAIParser, GameGraphParser

            ai_parser = LevelAIParser(ai_path, build_adjacency=False)
            ai_guid = ai_parser.guid

            gg = GameGraphParser.from_all_spawn(Path(all_spawn))
            cross_table = gg.get_cross_table_for_level(level_info['level_id'])

            if cross_table and 'level_guid' in cross_table:
                ct_guid = cross_table['level_guid']
                if ai_guid != ct_guid:
                    messagebox.showwarning(
                        "GUID Mismatch",
                        f"The level.ai for '{level_name}' has a different GUID than "
                        f"the cross table in all.spawn.\n\n"
                        f"This means the level.ai was rebuilt after the all.spawn was compiled. "
                        f"Vertex IDs may not correspond correctly.\n\n"
                        f"level.ai GUID:     {ai_guid.hex()}\n"
                        f"cross table GUID:  {ct_guid.hex()}"
                    )
        except Exception:
            pass  # Don't block if validation itself fails

        self._save_settings()

        # Build command - differs between frozen EXE and dev mode
        if getattr(sys, 'frozen', False):
            cmd = [sys.executable]
        else:
            cmd = [sys.executable, str(Path(__file__).resolve())]

        cmd += ["--visualise",
                "--ai-path", ai_path,
                "--level-id", str(level_info['level_id']),
                "--all-spawn", all_spawn]

        # Launch with stdout captured for progress display
        self._vis_proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                          text=True, bufsize=1)
        self._vis_queue = queue.Queue()
        self.view_level_btn.configure(state="disabled")
        self.vis_status_label.configure(text="Starting visualiser...")

        # Reader thread feeds lines into queue without blocking the GUI
        def _reader():
            for line in self._vis_proc.stdout:
                self._vis_queue.put(line)
            self._vis_queue.put(None)  # sentinel: EOF

        threading.Thread(target=_reader, daemon=True).start()
        self._poll_vis_output()

    def _poll_vis_output(self):
        """Poll visualiser subprocess stdout for loading progress."""
        loading_done = False
        try:
            while True:
                line = self._vis_queue.get_nowait()
                if line is None:
                    loading_done = True
                    break
                stripped = line.strip()
                if stripped:
                    self.vis_status_label.configure(text=stripped)
                    if stripped == "VISUALISER_READY":
                        loading_done = True
        except queue.Empty:
            pass

        ret = self._vis_proc.poll()
        if ret is not None or loading_done:
            self.vis_status_label.configure(text="")
            self.view_level_btn.configure(state="normal")
            self._vis_proc = None
        else:
            self.root.after(100, self._poll_vis_output)

    # --- Settings persistence ---

    def _load_settings(self):
        try:
            with open(SETTINGS_FILE, "r") as f:
                data = json.load(f)
            # Resolve to absolute (handles old relative settings gracefully)
            self.levels_dir_var.set(str(self._resolve_path(data.get("levels_dir", DEFAULTS["levels_dir"]))))
            self.output_dir_var.set(str(self._resolve_path(data.get("output_dir", DEFAULTS["output_dir"]))))
            override_dir = data.get("levels_override_dir", "")
            if override_dir:
                self.levels_override_dir_var.set(str(self._resolve_path(override_dir)))
            mod = data.get("base_mod", DEFAULTS["base_mod"])
            self.base_mod_var.set(mod if mod in MOD_FLAVOURS else DEFAULTS["base_mod"])
            geom = data.get("window_geometry", DEFAULTS["window_geometry"])
            self.root.geometry(geom)
            # Visualiser settings
            self.vis_spawn_var.set(data.get("vis_spawn_path", ""))
            self.vis_level_var.set(data.get("vis_level", ""))
        except (FileNotFoundError, json.JSONDecodeError, KeyError):
            self.root.geometry(DEFAULTS["window_geometry"])

        # Default all.spawn path if not set
        if not self.vis_spawn_var.get():
            output_dir = self._resolve_path(self.output_dir_var.get())
            self.vis_spawn_var.set(str(output_dir / "spawns" / "all.spawn"))

    def _save_settings(self):
        data = {
            "levels_dir": self.levels_dir_var.get(),
            "output_dir": self.output_dir_var.get(),
            "levels_override_dir": self.levels_override_dir_var.get(),
            "base_mod": self.base_mod_var.get(),
            "window_geometry": self.root.geometry(),
            "vis_spawn_path": self.vis_spawn_var.get(),
            "vis_level": self.vis_level_var.get(),
        }
        try:
            with open(SETTINGS_FILE, "w") as f:
                json.dump(data, f, indent=2)
        except Exception:
            pass

    # --- Directory browsing ---

    def _browse_dir(self, var: tk.StringVar):
        current = var.get()
        initial = Path(current) if Path(current).is_absolute() else PROJECT_ROOT / current
        if not initial.is_dir():
            initial = PROJECT_ROOT
        path = filedialog.askdirectory(initialdir=str(initial), title="Select Directory")
        if path:
            var.set(str(Path(path).resolve()))

    # --- Build execution ---

    def _resolve_path(self, path_str: str) -> Path:
        """Resolve a path that may be absolute or relative to project root."""
        p = Path(path_str)
        if p.is_absolute():
            return p
        return (PROJECT_ROOT / p).resolve()

    def _is_building(self) -> bool:
        return self.build_thread is not None and self.build_thread.is_alive()

    def _validate_before_build(self) -> bool:
        levels_path = self._resolve_path(self.levels_dir_var.get())
        if not levels_path.is_dir():
            messagebox.showerror("Error", f"Levels directory not found:\n{levels_path}")
            return False

        config_path = PROJECT_ROOT / "levels.ini"
        if not config_path.exists():
            messagebox.showerror("Error", f"levels.ini not found:\n{config_path}")
            return False

        return True

    def _set_controls_enabled(self, enabled: bool):
        state = "normal" if enabled else "disabled"
        self.levels_entry.configure(state=state)
        self.output_entry.configure(state=state)
        self.override_entry.configure(state=state)
        self.levels_browse_btn.configure(state=state)
        self.output_browse_btn.configure(state=state)
        self.override_browse_btn.configure(state=state)
        self.mod_combo.configure(state="readonly" if enabled else "disabled")
        # Disable/enable Visualiser tab
        self.notebook.tab(1, state="normal" if enabled else "disabled")

    def _on_full_build(self):
        if self._is_building():
            self._cancel_build()
            return

        if not self._validate_before_build():
            return

        self._save_settings()
        self._clear_output()
        self._start_build(deploy_only=False)

    def _on_deploy(self):
        if self._is_building():
            return

        if not self._validate_before_build():
            return

        self._save_settings()
        self._clear_output()
        self._start_build(deploy_only=True)

    def _start_build(self, deploy_only: bool = False):
        self.build_cancelled = False
        self.build_start_time = time.time()

        # Gather params for the build thread
        levels_dir = str(self._resolve_path(self.levels_dir_var.get()))
        output_dir = str(self._resolve_path(self.output_dir_var.get()))
        base_mod = self.base_mod_var.get()

        # Update UI
        self._set_controls_enabled(False)
        if deploy_only:
            self.full_build_btn.configure(state="disabled")
            self.deploy_btn.configure(state="disabled")
        else:
            self.full_build_btn.configure(text="Cancel Build", style="Cancel.TButton")
            self.deploy_btn.configure(state="disabled")

        # Start build thread
        self.build_thread = threading.Thread(
            target=self._run_build,
            args=(levels_dir, output_dir, base_mod, deploy_only),
            daemon=True,
        )
        self.build_thread.start()

        # Start polling
        self._poll_output()
        self._update_elapsed()

    def _run_build(self, levels_dir: str, output_dir: str, base_mod: str, deploy_only: bool):
        """Run the build in-process. Executes in a background thread."""
        # Add compiler/ to sys.path so its modules can be imported
        compiler_dir = str(COMPILER_DIR)
        if compiler_dir not in sys.path:
            sys.path.insert(0, compiler_dir)

        # Redirect stdout/stderr to the output queue
        stream = QueueStream(self.output_queue)
        old_stdout, old_stderr = sys.stdout, sys.stderr
        sys.stdout = stream
        sys.stderr = stream

        try:
            from project_paths import ProjectPaths
            from levels import LevelsConfig
            from utils import init_logging, print_summary, close_logging
            from build_all_spawn import GameGraphBuilder

            # Build central path config — all paths resolved once here
            levels_override = self.levels_override_dir_var.get() or None
            paths = ProjectPaths.from_root(
                PROJECT_ROOT,
                levels_dir=Path(levels_dir),
                output_dir=Path(output_dir),
                levels_override_dir=Path(levels_override) if levels_override else None,
            )

            init_logging(log_path=paths.output_dir / "build.log")

            config = LevelsConfig(
                config_path=str(paths.get_levels_ini()),
                levels_dir=paths.levels_dir,
                cross_table_dir=paths.build_dir,
                resolve_root=paths.compiler_dir,
                levels_override_dir=paths.levels_override_dir,
                build_dir=paths.build_dir,
            )

            builder = GameGraphBuilder(config, paths, base_mod=base_mod)
            builder.build_all(force_rebuild=False, deploy_only=deploy_only)

            close_logging()

        except Exception as e:
            print(f"\nERROR: {e}", file=stream)
            traceback.print_exc(file=stream)
        finally:
            sys.stdout = old_stdout
            sys.stderr = old_stderr
            self.output_queue.put(None)  # Sentinel: build finished

    def _poll_output(self):
        # Batch all pending output into a single widget update
        chunks = []
        finished = False
        try:
            while True:
                text = self.output_queue.get_nowait()
                if text is None:
                    finished = True
                    break
                chunks.append(text)
        except queue.Empty:
            pass

        if chunks:
            at_bottom = self.output_text.yview()[1] >= 0.99
            self.output_text.configure(state="normal")
            for text in chunks:
                clean = ANSI_RE.sub("", text)
                tag = self._detect_tag(clean)
                if tag:
                    self.output_text.insert("end", clean, tag)
                else:
                    self.output_text.insert("end", clean)
            self.output_text.configure(state="disabled")
            if at_bottom:
                self.output_text.see("end")

        if finished:
            self._on_build_finished()
            return

        # Keep polling until we receive the sentinel (None) from the build thread
        self.root.after(100, self._poll_output)

    def _update_elapsed(self):
        if self.build_start_time is None or not self._is_building():
            return
        elapsed = time.time() - self.build_start_time
        mins, secs = divmod(int(elapsed), 60)
        self.elapsed_label.configure(text=f"Elapsed: {mins}:{secs:02d}")
        self.elapsed_after_id = self.root.after(1000, self._update_elapsed)

    def _cancel_build(self):
        if not self._is_building():
            return
        self.build_cancelled = True
        self._append_output("\n--- Build cancellation requested. Will stop after current step. ---\n", tag="warning")

    def _on_build_finished(self):
        elapsed = time.time() - self.build_start_time if self.build_start_time else 0

        if self.build_cancelled:
            self._append_output("\n--- Build cancelled. ---\n", tag="warning")
        else:
            self._append_output(f"\nFinished in {elapsed:.1f}s.\n", tag="complete")

        self.build_thread = None
        self.build_start_time = None
        if self.elapsed_after_id:
            self.root.after_cancel(self.elapsed_after_id)
            self.elapsed_after_id = None

        # Restore UI
        self._set_controls_enabled(True)
        self.full_build_btn.configure(text="Full Build", style="FullBuild.TButton", state="normal")
        self.deploy_btn.configure(state="normal")

    # --- Output display ---

    def _clear_output(self):
        self.output_text.configure(state="normal")
        self.output_text.delete("1.0", "end")
        self.output_text.configure(state="disabled")

    def _append_output(self, text: str, tag: str = None):
        # Strip ANSI codes
        clean = ANSI_RE.sub("", text)

        # Auto-detect tag if not specified
        if tag is None:
            tag = self._detect_tag(clean)

        # Check if user has scrolled up
        at_bottom = self.output_text.yview()[1] >= 0.99

        self.output_text.configure(state="normal")
        if tag:
            self.output_text.insert("end", clean, tag)
        else:
            self.output_text.insert("end", clean)
        self.output_text.configure(state="disabled")

        if at_bottom:
            self.output_text.see("end")

    def _detect_tag(self, line: str) -> str:
        stripped = line.strip()
        if not stripped:
            return ""
        if stripped.startswith("ERROR:"):
            return "error"
        if stripped.startswith("Warning:"):
            return "warning"
        if re.match(r'^STEP \d', stripped) or re.match(r'^={10,}$', stripped):
            return "step"
        if "BUILD COMPLETE" in stripped or "FAST DEPLOY COMPLETE" in stripped:
            return "complete"
        if stripped.startswith("GAME GRAPH BUILDER") or stripped.startswith("FAST DEPLOY"):
            return "step"
        # Summary counts line: "N Error(s) | M Warning(s)"
        if "Error" in stripped and "Warning" in stripped and "|" in stripped:
            if not stripped.startswith("0 Error"):
                return "error"
            return "complete"
        # Override table entries
        if "OVERRIDE" in stripped:
            return "override"
        return ""

    # --- Window close ---

    def _on_close(self):
        if self._is_building():
            if not messagebox.askyesno("Build in progress",
                                       "A build is in progress. Exit anyway?"):
                return

        self._save_settings()
        self.root.destroy()


def _run_visualiser_cli():
    """Run the visualiser in a standalone process (launched via --visualise flag)."""
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--visualise", action="store_true")
    parser.add_argument("--ai-path", required=True)
    parser.add_argument("--level-id", type=int, required=True)
    parser.add_argument("--all-spawn", required=True)
    args = parser.parse_args()

    # In frozen EXE, bundled data files are extracted under sys._MEIPASS
    # In dev mode, they live relative to PROJECT_ROOT
    if getattr(sys, 'frozen', False):
        bundle_dir = Path(sys._MEIPASS)
    else:
        bundle_dir = PROJECT_ROOT

    # Add compiler paths first (for parsers, levels_config, etc.)
    compiler_dir = str(bundle_dir / "compiler")
    compiler_levels_dir = str(bundle_dir / "compiler" / "levels")
    for p in [compiler_dir, compiler_levels_dir]:
        if p not in sys.path:
            sys.path.insert(0, p)

    # Clear compiler's utils from module cache so visualiser's utils package loads
    if 'utils' in sys.modules:
        del sys.modules['utils']

    # Add visualiser dir at front so its packages (ui, utils, core) take priority
    vis_dir = str(bundle_dir / "visualiser")
    sys.path.insert(0, vis_dir)

    import open3d
    import open3d.visualization.gui as gui
    from ui import NodeInspectorApp

    # In a PyInstaller bundle, Open3D can't find its Filament resources
    # because the auto-detected path uses 8.3 short names on Windows.
    # Point it to the correct extracted location explicitly.
    if getattr(sys, 'frozen', False):
        o3d_dir = Path(open3d.__file__).parent
        resource_path = str(o3d_dir / "resources")
        gui.Application.instance.initialize(resource_path)
    else:
        gui.Application.instance.initialize()
    app = NodeInspectorApp(args.ai_path, level_id=args.level_id, all_spawn_path=args.all_spawn)

    # Set window icon (platform-specific, after window is created)
    _set_visualiser_icon()

    print("VISUALISER_READY", flush=True)
    app.run()


def _set_visualiser_icon():
    """Set the Open3D window icon to the LAMBDA icon."""
    if platform.system() == "Windows":
        ico_path = str(ASSETS_DIR / "docs" / "images" / "visualiser_icon.ico")
        if not Path(ico_path).exists():
            return
        try:
            import ctypes
            user32 = ctypes.windll.user32
            hwnd = user32.FindWindowW(None, "Leyths' Level Vertex Graph Inspector")
            if not hwnd:
                return
            IMAGE_ICON = 1
            LR_LOADFROMFILE = 0x00000010
            LR_DEFAULTSIZE = 0x00000040
            WM_SETICON = 0x0080
            ICON_BIG = 1
            ICON_SMALL = 0
            hicon_big = user32.LoadImageW(None, ico_path, IMAGE_ICON, 0, 0,
                                          LR_LOADFROMFILE | LR_DEFAULTSIZE)
            hicon_small = user32.LoadImageW(None, ico_path, IMAGE_ICON, 16, 16,
                                            LR_LOADFROMFILE)
            if hicon_big:
                user32.SendMessageW(hwnd, WM_SETICON, ICON_BIG, hicon_big)
            if hicon_small:
                user32.SendMessageW(hwnd, WM_SETICON, ICON_SMALL, hicon_small)
        except Exception:
            pass
    elif platform.system() == "Darwin":
        icon_file = ASSETS_DIR / "docs" / "images" / "visualiser_icon.png"
        if not icon_file.exists():
            return
        try:
            from AppKit import NSApplication, NSImage
            ns_app = NSApplication.sharedApplication()
            ns_icon = NSImage.alloc().initWithContentsOfFile_(str(icon_file))
            ns_app.setApplicationIconImage_(ns_icon)
        except ImportError:
            pass


def main(build_overrides=None):
    # Close PyInstaller splash screen if present
    try:
        import pyi_splash
        pyi_splash.close()
    except ImportError:
        pass

    root = tk.Tk()
    gui = LambdaGUI(root, build_overrides=build_overrides)
    root.mainloop()


def _parse_build_args():
    """Parse --build CLI args and return overrides dict, or None if not a --build invocation."""
    import argparse
    parser = argparse.ArgumentParser(description="LAMBDA build")
    parser.add_argument("--build", action="store_true")
    parser.add_argument("--levels-dir", default=None)
    parser.add_argument("--levels-override-dir", default=None)
    parser.add_argument("--output-dir", default=None)
    parser.add_argument("--basemod", default=None)
    parser.add_argument("--deploy-only", action="store_true")
    args = parser.parse_args()

    overrides = {}
    if args.levels_dir:
        overrides["levels_dir"] = args.levels_dir
    if args.levels_override_dir:
        overrides["levels_override_dir"] = args.levels_override_dir
    if args.output_dir:
        overrides["output_dir"] = args.output_dir
    if args.basemod:
        overrides["base_mod"] = args.basemod
    overrides["deploy_only"] = args.deploy_only
    return overrides


if __name__ == "__main__":
    import multiprocessing
    multiprocessing.freeze_support()
    if "--visualise" in sys.argv:
        _run_visualiser_cli()
    elif "--build" in sys.argv:
        overrides = _parse_build_args()
        main(build_overrides=overrides)
    else:
        main()

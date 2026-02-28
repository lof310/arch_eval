"""Visualization utilities: real-time window, video recording, plot saving."""

import os
import logging
import subprocess
import tempfile
import shutil
import threading
import queue
import atexit
from collections import deque, defaultdict
from typing import Dict, List, Optional, Any

import numpy as np
import psutil
import torch

logger = logging.getLogger(__name__)


class PlotSaver:
    """Saves final plots of metrics."""

    def __init__(self, config, history: Dict[str, List[float]]):
        self.config = config
        self.history = history

    def save_plots(self, out_dir: str = "./plots"):
        """Save plots for specified metrics."""
        # Import matplotlib only when needed and use Agg backend
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(out_dir, exist_ok=True)
        to_plot = self.config.save_plot or list(self.history.keys())

        for metric in to_plot:
            # Buscar la métrica con o sin prefijo
            data = None
            if metric in self.history:
                data = self.history[metric]
            elif f"train_{metric}" in self.history:
                data = self.history[f"train_{metric}"]
            elif f"val_{metric}" in self.history:
                data = self.history[f"val_{metric}"]

            if data is None or not data:
                logger.warning(f"No data for metric: {metric}")
                continue

            steps = range(len(data))
            plt.figure()
            sns.lineplot(x=steps, y=data, label=metric)

            if len(data) > 10:
                try:
                    z = np.polyfit(steps, data, 1)
                    p = np.poly1d(z)
                    plt.plot(steps, p(steps), "--", alpha=0.7, label="trend")
                except Exception:
                    pass

            plt.title(f"{metric} over time")
            plt.xlabel("Step")
            plt.ylabel("Value")
            plt.legend()
            safe = metric.replace("/", "_").replace(" ", "_")
            plt.savefig(os.path.join(out_dir, f"{safe}.png"), dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved: {safe}.png")


class VideoRecorder:
    """Records metrics over time and generates video using ffmpeg."""

    def __init__(self, config, metrics: List[str], history_len: int = 100,
                 fps: int = 10, codec: str = "libx264", resolution: tuple = None):
        self.config = config
        self.metrics = metrics
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.frames_dir = {}
        self.frame_counts = defaultdict(int)
        self.steps = []
        self.histories = {m: deque(maxlen=history_len) for m in metrics}

        # Usar TemporaryDirectory para limpieza automática
        self.base_temp = tempfile.mkdtemp(prefix="arch_eval_video_")
        self._temp_dir_created = True

        for m in metrics:
            d = os.path.join(self.base_temp, m.replace("/", "_"))
            os.makedirs(d, exist_ok=True)
            self.frames_dir[m] = d

    def record_step(self, step: int, metrics: Dict[str, float]):
        """Record metrics for current step and save a frame."""
        self.steps.append(step)
        for m in self.metrics:
            if m in metrics:
                val = metrics[m]
                self.histories[m].append((step, val))
                frame = self._create_frame(m, self.histories[m], step)
                self._save_frame(m, frame)

    def _create_frame(self, metric, history, current_step):
        """Create a single frame for a metric."""
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), 
                                        gridspec_kw={"height_ratios": [3, 1]})
        if history:
            steps, vals = zip(*history)
            ax1.plot(steps, vals, "b-", linewidth=2)
            ax1.set_xlabel("Step")
            ax1.set_ylabel(metric)
            ax1.set_title(f"{metric} over time")
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=current_step, color="r", linestyle="--", alpha=0.7)

        cur = history[-1][1] if history else 0
        ax2.text(0.5, 0.5, f"Current: {cur:.4f}", ha="center", va="center", 
                transform=ax2.transAxes, fontsize=16, 
                bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5))
        ax2.axis("off")

        plt.tight_layout()
        fig.canvas.draw()

        # Convert to numpy array
        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return frame

    def _save_frame(self, metric, frame):
        """Save frame to disk."""
        path = os.path.join(self.frames_dir[metric], f"frame_{self.frame_counts[metric]:06d}.png")
        import matplotlib.pyplot as plt
        plt.imsave(path, frame)
        self.frame_counts[metric] += 1

    def save_video(self, out_path: str):
        """Generate video from recorded frames using ffmpeg."""
        if not self.frames_dir:
            logger.warning("No frames recorded, skipping video")
            return

        try:
            for metric, d in self.frames_dir.items():
                if self.frame_counts[metric] == 0:
                    continue

                video = f"{out_path}_{metric}.mp4"
                cmd = ["ffmpeg", "-y", "-framerate", str(self.fps), 
                       "-pattern_type", "glob", "-i", os.path.join(d, "*.png"), 
                       "-c:v", self.codec, "-pix_fmt", "yuv420p"]

                if self.resolution:
                    cmd.extend(["-s", f"{self.resolution[0]}x{self.resolution[1]}"])

                cmd.append(video)

                logger.info(f"Generating video for {metric}...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if res.returncode != 0:
                    logger.error(f"FFmpeg error for {metric}: {res.stderr}")
                else:
                    logger.info(f"Video saved to {video}")

        except FileNotFoundError:
            logger.error("ffmpeg not found. Please install ffmpeg.")
        except Exception as e:
            logger.error(f"Error creating video: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Remove temporary frame files."""
        if hasattr(self, 'base_temp') and os.path.exists(self.base_temp):
            shutil.rmtree(self.base_temp, ignore_errors=True)


class RealtimeWindow:
    """Non-interactive window displaying real-time metrics and system resources."""

    def __init__(self, config, max_points: int = 1000, figsize: tuple = (14, 8)):
        self.config = config
        self.max_points = max_points
        self.figsize = figsize
        self.metrics_history = deque(maxlen=max_points)
        self.system_history = deque(maxlen=max_points)
        self.step_counter = 0
        self.update_queue = queue.Queue()
        self.stop_event = threading.Event()
        self.disabled = False
        self.root = None
        self.running = False

        try:
            # Intentar importar tkinter y configurar matplotlib
            import matplotlib
            matplotlib.use('TkAgg')
            import tkinter as tk
            from tkinter import ttk
            from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
            from matplotlib.figure import Figure

            self.tk = tk
            self.ttk = ttk
            self.FigureCanvasTkAgg = FigureCanvasTkAgg
            self.Figure = Figure

            # Iniciar ventana en hilo separado
            self.thread = threading.Thread(target=self._run_window, daemon=True)
            self.thread.start()
            atexit.register(self.close)

        except Exception as e:
            logger.warning(f"Could not initialize realtime window: {e}")
            self.disabled = True

    def _run_window(self):
        """Run the tkinter main loop in separate thread."""
        try:
            self.root = self.tk.Tk()
            self.root.title("arch_eval - Training Monitor")
            self.root.protocol("WM_DELETE_WINDOW", self._on_closing)
            self.running = True

            main = self.ttk.Frame(self.root)
            main.pack(fill=self.tk.BOTH, expand=True)

            self.notebook = self.ttk.Notebook(main)
            self.notebook.pack(fill=self.tk.BOTH, expand=True)

            # Metrics tab
            metrics_frame = self.ttk.Frame(self.notebook)
            self.notebook.add(metrics_frame, text="Metrics")
            self._setup_metrics_plot(metrics_frame)

            # System tab
            system_frame = self.ttk.Frame(self.notebook)
            self.notebook.add(system_frame, text="System")
            self._setup_system_plot(system_frame)

            # Model info tab (optional)
            info_frame = self.ttk.Frame(self.notebook)
            self.notebook.add(info_frame, text="Model Info")
            self._setup_info_tab(info_frame)

            self._check_queue()
            self.root.mainloop()

        except Exception as e:
            logger.error(f"Error in realtime window thread: {e}")
            self.running = False

    def _setup_metrics_plot(self, parent):
        """Setup metrics plotting area."""
        n_metrics = len(self.config.save_video) if self.config.save_video else 4
        n_cols = min(2, n_metrics)
        n_rows = (n_metrics + n_cols - 1) // n_cols

        self.fig = self.Figure(figsize=self.figsize)
        self.axes = {}

        for i in range(n_metrics):
            ax = self.fig.add_subplot(n_rows, n_cols, i + 1)
            name = self.config.save_video[i] if i < len(self.config.save_video) else f"Metric {i+1}"
            self.axes[name] = ax
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)

        self.canvas = self.FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=self.tk.BOTH, expand=True)

    def _setup_system_plot(self, parent):
        """Setup system resource monitoring plot."""
        sys_fig = self.Figure(figsize=(10, 6))
        self.system_axes = sys_fig.add_subplot(111)
        self.system_axes.set_title("System Resources")
        self.system_axes.set_xlabel("Step")
        self.system_axes.set_ylabel("Usage %")
        self.system_axes.grid(True, alpha=0.3)

        self.system_canvas = self.FigureCanvasTkAgg(sys_fig, master=parent)
        self.system_canvas.draw()
        self.system_canvas.get_tk_widget().pack(fill=self.tk.BOTH, expand=True)

    def _setup_info_tab(self, parent):
        """Setup model info tab."""
        label = self.ttk.Label(parent, text="Model information will appear here.\n(Learning rate, gradient norm, etc.)")
        label.pack(padx=10, pady=10)

    def _check_queue(self):
        """Check for updates from training thread."""
        if not self.running:
            return

        try:
            while True:
                upd = self.update_queue.get_nowait()
                self._process_update(upd)
        except queue.Empty:
            pass

        if not self.stop_event.is_set() and self.running:
            self.root.after(100, self._check_queue)

    def _process_update(self, update: Dict[str, Any]):
        """Process an update from training."""
        self.metrics_history.append(update)
        self.system_history.append(self._get_system_stats())
        self._update_metrics_plots()
        self._update_system_plot()

        if self.canvas:
            self.canvas.draw_idle()
        if self.system_canvas:
            self.system_canvas.draw_idle()

    def _get_system_stats(self) -> Dict[str, float]:
        """Get current system resource usage."""
        stats = {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent
        }

        if torch.cuda.is_available():
            stats["gpu_percent"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
            stats["gpu_memory"] = torch.cuda.memory_allocated() / 1024**3

        return stats

    def _update_metrics_plots(self):
        """Update all metric plots with latest data."""
        if not self.metrics_history:
            return

        for ax in self.axes.values():
            ax.clear()

        for name, ax in self.axes.items():
            vals = [m.get(name, 0) for m in self.metrics_history]
            steps = range(len(vals))
            ax.plot(steps, vals, label=name)
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            ax.legend()

    def _update_system_plot(self):
        """Update system resource plot."""
        if not self.system_history:
            return

        self.system_axes.clear()

        cpu = [s["cpu_percent"] for s in self.system_history]
        mem = [s["memory_percent"] for s in self.system_history]
        steps = range(len(cpu))

        self.system_axes.plot(steps, cpu, label="CPU %", color="blue")
        self.system_axes.plot(steps, mem, label="Memory %", color="green")

        if "gpu_percent" in self.system_history[0]:
            gpu = [s.get("gpu_percent", 0) for s in self.system_history]
            self.system_axes.plot(steps, gpu, label="GPU %", color="red")

        self.system_axes.set_title("System Resources")
        self.system_axes.set_xlabel("Step")
        self.system_axes.set_ylabel("Usage %")
        self.system_axes.grid(True, alpha=0.3)
        self.system_axes.legend()

    def update(self, metrics: Dict[str, float]):
        """Public method to update window with new metrics."""
        if self.disabled:
            return

        self.step_counter += 1
        if self.step_counter % self.config.viz_interval == 0:
            self.update_queue.put(metrics)

    def _on_closing(self):
        """Handle window closing."""
        self.stop_event.set()
        self.running = False
        if self.root:
            try:
                self.root.quit()
            except:
                pass

    def close(self):
        """Close the window safely."""
        if self.disabled:
            return
        self.stop_event.set()
        self.running = False

        if self.root:
            try:
                self.root.after_idle(self._safe_destroy)
            except:
                pass

    def _safe_destroy(self):
        """Destroy the root window safely."""
        try:
            if self.root and self.root.winfo_exists():
                self.root.quit()
                self.root.destroy()
        except:
            pass

    def __del__(self):
        """Cleanup on deletion."""
        self.close()

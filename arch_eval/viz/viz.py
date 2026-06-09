"""Visualization utilities: real-time window, video recording and plot saving."""

import logging
import os
import queue
import shutil
import subprocess
import tempfile
import threading
from collections import defaultdict, deque
from typing import Any, Dict, List, Optional

import numpy as np
import torch

logger = logging.getLogger(__name__)


def _gui_available() -> bool:
    """Check if a GUI display is available for matplotlib/TkAgg backend."""
    try:
        import tkinter

        root = tkinter.Tk()
        root.destroy()  # Destroy the window immediately to avoid zombie processes
        return True
    except Exception:
        return False


class TerminalProgress:
    """Terminal-based progress display using rich/tqdm with graceful fallbacks."""

    def __init__(self, config, metric_names: Optional[List[str]] = None):
        self.config = config
        self.metric_names = metric_names or []
        self.metrics_history = {}
        self.best_metrics = {}
        self.step_counter = 0
        self.disabled = False
        # Try importing rich, then tqdm, then fall back to plain print
        self._backend = "plain"
        try:
            from rich.console import Console
            from rich.live import Live
            from rich.panel import Panel
            from rich.table import Table

            self.Console = Console
            self.Live = Live
            self.Table = Table
            self.Panel = Panel
            self._backend = "rich"
        except ImportError:
            try:
                from tqdm import tqdm

                self.tqdm = tqdm
                self._backend = "tqdm"
            except ImportError:
                logger.info("Neither rich nor tqdm available, using plain terminal output")
        if self._backend == "rich":
            self.console = self.Console()
            self.live = None
        elif self._backend == "tqdm":
            self.pbar = None

    def start(self, total_steps: Optional[int] = None):
        """Start the progress display."""
        if self._backend == "rich":
            self.live = self.Live(self._make_table({}), console=self.console, refresh_per_second=4)
            self.live.start()
        elif self._backend == "tqdm":
            self.pbar = self.tqdm(total=total_steps, desc="Training")

    def _make_table(self, metrics: Dict[str, float]) -> Any:
        """Create a table/panel with current metrics."""
        if self._backend == "rich":
            table = self.Table(title="Training Progress", expand=True)
            table.add_column("Metric", style="cyan")
            table.add_column("Current", style="green")
            table.add_column("Best", style="yellow")
            for name in self.metric_names[:6]:
                if name in metrics:
                    val = metrics[name]
                    best = self.best_metrics.get(name, val)
                    if "loss" in name.lower():
                        if val < best:
                            self.best_metrics[name] = val
                            best = val
                    else:
                        if val > best:
                            self.best_metrics[name] = val
                            best = val
                    table.add_row(name, f"{val:.4f}", f"{best:.4f}")
            from arch_eval._lazy import lazy_import

            psutil = lazy_import("psutil")
            cpu = psutil.cpu_percent()
            mem = psutil.virtual_memory().percent
            gpu_mem = ""
            if torch.cuda.is_available():
                gpu_mem = f"{torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100:.1f}%"
            table.add_row("CPU %", f"{cpu:.1f}", "")
            table.add_row("Memory %", f"{mem:.1f}", "")
            if gpu_mem:
                table.add_row("GPU Mem %", gpu_mem, "")
            return self.Panel(table, title=f"Step {self.step_counter}")
        else:
            return metrics

    def update(self, metrics: Dict[str, float]):
        """Update the display with new metrics."""
        if self.disabled:
            return
        self.step_counter += 1
        # Update history regardless of viz_interval
        for name, value in metrics.items():
            if name not in self.metrics_history:
                self.metrics_history[name] = deque(maxlen=100)
            self.metrics_history[name].append(value)
        # Update tqdm progress bar every step with postfix
        if self._backend == "tqdm" and self.pbar:
            self.pbar.update(1)
            desc_parts = [f"{k}: {v:.4f}" for k, v in list(metrics.items())[:3]]
            self.pbar.set_postfix_str(", ".join(desc_parts))
        elif self._backend == "rich" and self.live:
            # Update rich display every step
            self.live.update(self._make_table(metrics))
        else:
            # Plain backend: print single line with \r
            if self.step_counter % (self.config.log_interval * 10) == 0:
                parts = [f"{k}: {v:.4f}" for k, v in metrics.items()]
                print(f"\rStep {self.step_counter}: " + ", ".join(parts[:5]), end="", flush=True)

    def close(self):
        """Close the progress display."""
        if self._backend == "rich" and self.live:
            self.live.stop()
        elif self._backend == "tqdm" and self.pbar:
            self.pbar.close()
        elif self._backend == "plain":
            print()  # Newline after plain output


class PlotSaver:
    """Saves final plots of metrics."""

    def __init__(self, config, history: Dict[str, List[float]]):
        self.config = config
        self.history = history

    def save_plots(self, out_dir: str = "./plots"):
        """Save plots for specified metrics."""
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(out_dir, exist_ok=True)
        to_plot = self.config.save_plot or list(self.history.keys())

        for metric in to_plot:
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
            base_path = os.path.join(out_dir, safe)
            plt.savefig(f"{base_path}.png", dpi=150, bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved: {safe}.png")


class VideoRecorder:
    """Records metrics over time and generates video using ffmpeg."""

    def __init__(
        self,
        config,
        metrics: List[str],
        history_len: int = 100,
        fps: int = 10,
        codec: str = "libx264",
        resolution: tuple = None,
    ):
        self.config = config
        self.metrics = metrics
        self.fps = fps
        self.codec = codec
        self.resolution = resolution
        self.frames_dir = {}
        self.frame_counts = defaultdict(int)
        self.steps = []
        self.histories = {m: deque(maxlen=history_len) for m in metrics}

        # Check ffmpeg availability
        self.ffmpeg_available = self._check_ffmpeg()

        # Defer temp directory creation until first record_step
        self.base_temp = None

    def _initialize_temp(self):
        """Initialize temporary directory for frames (called on first record_step)."""
        if self.base_temp is not None:
            return
        self.base_temp = tempfile.mkdtemp(prefix="arch_eval_video_")
        for m in self.metrics:
            d = os.path.join(self.base_temp, m.replace("/", "_"))
            os.makedirs(d, exist_ok=True)
            self.frames_dir[m] = d

    def _check_ffmpeg(self):
        """Check if ffmpeg is installed and accessible."""
        try:
            subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)
            return True
        except (subprocess.SubprocessError, FileNotFoundError):
            logger.warning("ffmpeg not found. Video recording disabled.")
            return False

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

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]})
        if history:
            steps, vals = zip(*history)
            ax1.plot(steps, vals, "b-", linewidth=2)
            ax1.set_xlabel("Step")
            ax1.set_ylabel(metric)
            ax1.set_title(f"{metric} over time")
            ax1.grid(True, alpha=0.3)
            ax1.axvline(x=current_step, color="r", linestyle="--", alpha=0.7)

        cur = history[-1][1] if history else 0
        ax2.text(
            0.5,
            0.5,
            f"Current: {cur:.4f}",
            ha="center",
            va="center",
            transform=ax2.transAxes,
            fontsize=16,
            bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
        )
        ax2.axis("off")

        plt.tight_layout()
        fig.canvas.draw()

        frame = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
        frame = frame.reshape(fig.canvas.get_width_height()[::-1] + (3,))

        plt.close(fig)
        return frame

    def _save_frame(self, metric, frame):
        """Save frame to disk."""
        import matplotlib.pyplot as plt

        path = os.path.join(self.frames_dir[metric], f"frame_{self.frame_counts[metric]:06d}.png")
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
                pattern = os.path.join(d, "*.png")
                cmd = [
                    "ffmpeg",
                    "-y",
                    "-framerate",
                    str(self.fps),
                    "-pattern_type",
                    "glob",
                    "-i",
                    pattern,
                    "-c:v",
                    self.codec,
                    "-pix_fmt",
                    "yuv420p",
                ]

                if self.resolution:
                    cmd.extend(["-s", f"{self.resolution[0]}x{self.resolution[1]}"])

                cmd.append(video)

                logger.info(f"Generating video for {metric}...")
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=300)

                if res.returncode != 0:
                    logger.error(f"FFmpeg error for {metric}: {res.stderr}")
                else:
                    logger.info(f"Video saved to {video}")

        except Exception as e:
            logger.error(f"Error creating video: {e}")
        finally:
            self._cleanup()

    def _cleanup(self):
        """Remove temporary frame files."""
        if hasattr(self, "base_temp") and os.path.exists(self.base_temp):
            shutil.rmtree(self.base_temp, ignore_errors=True)


class RealtimeWindow:
    """Non-interactive window displaying real-time metrics and system resources."""

    def __init__(
        self, config, metric_names: Optional[List[str]] = None, max_points: int = 1000, figsize: tuple = (14, 8)
    ):
        self.config = config
        self.max_points = max_points
        self.figsize = figsize
        self.metric_names = metric_names or []
        self.metrics_history = {}  # dict of deques keyed by metric name
        self.system_history = deque(maxlen=max_points)
        self.step_counter = 0
        self.disabled = False

        # Pre-check GUI availability before attempting to create figure
        if not _gui_available():
            logger.warning("GUI display not available. RealtimeWindow disabled.")
            self.disabled = True
            return

        try:
            import matplotlib

            matplotlib.use("TkAgg")
            import matplotlib.pyplot as plt

            self.plt = plt
            self.plt.ion()
            self._setup_plots()
            # Set window title and ensure initial draw
            self.fig.canvas.manager.set_window_title("arch_eval Training Monitor")
            self.fig.canvas.draw_idle()
            self.plt.pause(0.001)
            self.plt.show(block=False)
        except Exception as e:
            logger.warning(f"Could not initialize realtime window: {e}")
            self.disabled = True

    def _setup_plots(self):
        # We'll have up to 4 metric subplots + 1 system subplot
        self.max_metric_plots = 4
        self.n_plots = self.max_metric_plots + 1
        self.n_cols = 2
        self.n_rows = (self.n_plots + self.n_cols - 1) // self.n_cols

        self.fig, self.axes = self.plt.subplots(self.n_rows, self.n_cols, figsize=self.figsize, squeeze=False)
        self.axes = self.axes.flatten()

        for ax in self.axes:
            ax.set_visible(False)

        self.system_ax = self.axes[-1]
        self.system_ax.set_visible(True)
        self.system_ax.set_title("System Resources")
        self.system_ax.set_xlabel("Step")
        self.system_ax.set_ylabel("Usage %")
        self.system_ax.grid(True, alpha=0.3)
        (self.cpu_line,) = self.system_ax.plot([], [], label="CPU %", color="blue")
        (self.mem_line,) = self.system_ax.plot([], [], label="Memory %", color="green")
        if torch.cuda.is_available():
            (self.gpu_line,) = self.system_ax.plot([], [], label="GPU %", color="red")
        else:
            self.gpu_line = None
        self.system_ax.legend()

        self.metric_lines = {}
        self.metric_axes = {}
        self.next_metric_idx = 0

    def _get_system_stats(self):
        from arch_eval._lazy import lazy_import

        psutil = lazy_import("psutil")
        stats = {"cpu": psutil.cpu_percent(), "memory": psutil.virtual_memory().percent}
        if torch.cuda.is_available():
            stats["gpu"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        return stats

    def _add_metric(self, name):
        if self.next_metric_idx >= self.max_metric_plots:
            logger.warning(
                f"RealtimeWindow: maximum number of metrics ({self.max_metric_plots}) reached, ignoring {name}"
            )
            return
        ax = self.axes[self.next_metric_idx]
        ax.set_visible(True)
        ax.set_title(name)
        ax.set_xlabel("Step")
        ax.set_ylabel("Value")
        ax.grid(True, alpha=0.3)
        (line,) = ax.plot([], [], lw=2)
        self.metric_lines[name] = line
        self.metric_axes[name] = ax
        self.metrics_history[name] = deque(maxlen=self.max_points)
        self.next_metric_idx += 1
        # Adjust layout
        self.fig.tight_layout()

    def update(self, metrics: Dict[str, float]):
        if self.disabled:
            return
        self.step_counter += 1
        # Reduce CPU usage by plotting less frequently
        plot_every = max(1, self.config.viz_interval // 2)
        if self.step_counter % plot_every != 0:
            return

        self.system_history.append(self._get_system_stats())

        for name, value in metrics.items():
            if name not in self.metrics_history:
                if self.next_metric_idx < self.max_metric_plots:
                    self._add_metric(name)
                else:
                    continue
            self.metrics_history[name].append(value)

        self._update_plots()

    def _update_plots(self):
        for name, line in self.metric_lines.items():
            data = list(self.metrics_history.get(name, []))
            if data:
                steps = list(range(len(data)))
                line.set_data(steps, data)
                ax = self.metric_axes[name]
                ax.relim()
                ax.autoscale_view()

        if self.system_history:
            steps = range(len(self.system_history))
            cpu_vals = [s["cpu"] for s in self.system_history]
            mem_vals = [s["memory"] for s in self.system_history]
            self.cpu_line.set_data(steps, cpu_vals)
            self.mem_line.set_data(steps, mem_vals)
            if self.gpu_line and "gpu" in self.system_history[0]:
                gpu_vals = [s.get("gpu", 0) for s in self.system_history]
                self.gpu_line.set_data(steps, gpu_vals)
            self.system_ax.relim()
            self.system_ax.autoscale_view()

        self.fig.canvas.draw_idle()
        self.plt.pause(0.001)
        self.fig.canvas.flush_events()

    def close(self):
        if self.disabled:
            return
        self.plt.close(self.fig)

"""Visualization utilities: real-time window, video recording and plot saving."""

import atexit
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

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import seaborn as sns

        os.makedirs(out_dir, exist_ok=True)
        to_plot = self.config.save_plot or list(self.history.keys())

        for metric in to_plot:
            # Search the metric with or without prefix
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
            if fmt == "pdf" or fmt == "both":
                plt.savefig(f"{base_path}.pdf", bbox_inches="tight")
            plt.close()
            logger.info(f"Plot saved: {safe}.{fmt}")

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

        self.base_temp = tempfile.mkdtemp(prefix="arch_eval_video_")
        self._temp_dir_created = True

        for m in metrics:
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
                pattern = os.path.join(d, "*.png")
                cmd = ["ffmpeg", "-y", "-framerate", str(self.fps), 
                       "-pattern_type", "glob", "-i", pattern, 
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
            import matplotlib
            matplotlib.use('TkAgg')  # or 'Qt5Agg' depending on system
            import matplotlib.pyplot as plt
            self.plt = plt
            self.plt.ion()  # interactive mode on
            self._setup_plots()
            self.plt.show(block=False)
            self.plt.pause(0.001)
        except Exception as e:
            logger.warning(f"Could not initialize realtime window: {e}")
            self.disabled = True

    def _setup_plots(self):
        n_metrics = len(self.metrics_history)
        n_system = 1  # system stats
        total_plots = n_metrics + n_system
        n_cols = min(2, total_plots)
        n_rows = (total_plots + n_cols - 1) // n_cols

        self.fig, self.axes = self.plt.subplots(n_rows, n_cols, figsize=self.figsize, squeeze=False)
        self.axes = self.axes.flatten()

        # Metrics subplots
        self.metric_lines = {}
        for i, name in enumerate(self.metrics_history.keys()):
            ax = self.axes[i]
            ax.set_title(name)
            ax.set_xlabel("Step")
            ax.set_ylabel("Value")
            ax.grid(True, alpha=0.3)
            line, = ax.plot([], [], lw=2, label=name)
            self.metric_lines[name] = line
            ax.legend()

        # System stats subplot
        sys_ax = self.axes[n_metrics]
        sys_ax.set_title("System Resources")
        sys_ax.set_xlabel("Step")
        sys_ax.set_ylabel("Usage %")
        sys_ax.grid(True, alpha=0.3)
        self.cpu_line, = sys_ax.plot([], [], label="CPU %", color="blue")
        self.mem_line, = sys_ax.plot([], [], label="Memory %", color="green")
        if torch.cuda.is_available():
            self.gpu_line, = sys_ax.plot([], [], label="GPU %", color="red")
        else:
            self.gpu_line = None
        sys_ax.legend()

        # Hide unused subplots
        for j in range(total_plots, len(self.axes)):
            self.axes[j].set_visible(False)

        self.fig.tight_layout()

    def _get_system_stats(self):
        stats = {
            "cpu": psutil.cpu_percent(),
            "memory": psutil.virtual_memory().percent
        }
        if torch.cuda.is_available():
            stats["gpu"] = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() * 100
        return stats

    def _animate(self, frame):
        # Update metric lines
        for name, line in self.metric_lines.items():
            data = list(self.metrics_history[name])
            if data:
                line.set_data(range(len(data)), data)
                ax = line.axes
                ax.relim()
                ax.autoscale_view()

        # Update system plot
        if self.system_history:
            steps = range(len(self.system_history))
            cpu_vals = [s["cpu"] for s in self.system_history]
            mem_vals = [s["memory"] for s in self.system_history]
            self.cpu_line.set_data(steps, cpu_vals)
            self.mem_line.set_data(steps, mem_vals)
            if self.gpu_line and "gpu" in self.system_history[0]:
                gpu_vals = [s.get("gpu", 0) for s in self.system_history]
                self.gpu_line.set_data(steps, gpu_vals)
            ax = self.cpu_line.axes
            ax.relim()
            ax.autoscale_view()

        return list(self.metric_lines.values()) + [self.cpu_line, self.mem_line] + ([self.gpu_line] if self.gpu_line else [])

    def update(self, metrics: Dict[str, float]):
        if self.disabled:
            return
        self.step_counter += 1
        if self.step_counter % self.config.viz_interval != 0:
            return
        for name in self.metrics_history:
            if name in metrics:
                self.metrics_history[name].append(metrics[name])
        self.system_history.append(self._get_system_stats())

    def close(self):
        if self.disabled:
            return
        self.plt.close(self.fig)

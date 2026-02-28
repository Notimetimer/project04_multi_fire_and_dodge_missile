import sys
import os
import time
import threading
import subprocess
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime
import numpy as np

class TensorBoardLogger:
    def __init__(self, log_root="./logs", host="127.0.0.1", port=6006, use_log_root=False, auto_show=False):
        self.log_root = os.path.abspath(log_root)
        os.makedirs(self.log_root, exist_ok=True)
        if use_log_root:
            self.run_dir = log_root
        else:
            self.run_dir = os.path.join(self.log_root, "run-" + datetime.now().strftime("%Y%m%d-%H%M%S"))
        os.makedirs(self.run_dir, exist_ok=True)
        self.writer = SummaryWriter(self.run_dir)
        self.tb_proc = None
        self.host = host
        self.port = port
        self.auto_show = auto_show
        if self.auto_show:
            self._start_tensorboard()

    def _read_stream(self, stream, prefix):
        for line in iter(stream.readline, ""):
            if not line:
                break
            print(f"{prefix}: {line.rstrip()}")

    def _start_tensorboard(self):
        cmd = [
            sys.executable,
            "-m",
            "tensorboard.main",
            f"--logdir={self.run_dir}",
            f"--host={self.host}",
            f"--port={self.port}"
        ]
        try:
            self.tb_proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1
            )
            threading.Thread(target=self._read_stream, args=(self.tb_proc.stdout, "TB-OUT"), daemon=True).start()
            threading.Thread(target=self._read_stream, args=(self.tb_proc.stderr, "TB-ERR"), daemon=True).start()
            print(f"TensorBoard 启动成功，访问 http://{self.host}:{self.port}，日志目录：{self.run_dir}")
        except Exception as e:
            print("启动 TensorBoard 失败：", e)
            self.tb_proc = None

    def add(self, tag, value, step):
        self.writer.add_scalar(tag, value, step)
        self.writer.flush()

    def close(self):
        try:
            self.writer.flush()
        except Exception:
            pass
        try:
            self.writer.close()
        except Exception:
            pass
        if self.tb_proc is not None:
            try:
                self.tb_proc.terminate()
                self.tb_proc.wait(timeout=3)
            except Exception:
                try:
                    self.tb_proc.kill()
                except Exception:
                    pass

# 示例用法
if __name__ == "__main__":
    logger = TensorBoardLogger(log_root="./logs", host="127.0.0.1", port=6006, auto_show=False)
    try:
        step = 0
        while True:
            logger.add("sin", np.sin(step * np.pi / 180.0), step)
            logger.add("cos", np.cos(step * np.pi / 180.0), step)
            logger.add("exp", np.exp(step / 360.0), step)
            print(f"[data] step={step} sin={np.sin(step * np.pi / 180.0):.4f} cos={np.cos(step * np.pi / 180.0):.4f} exp={np.exp(step / 360.0):.4f}")
            step += 1
            time.sleep(0.3)
    except KeyboardInterrupt:
        print("\n检测到 KeyboardInterrupt，正在关闭 logger ...")
    finally:
        logger.close()
        print(f"日志已保存到：{logger.run_dir}")


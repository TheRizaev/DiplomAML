# -*- coding: utf-8 -*-
"""Шаг 5a. Проверка компьютера перед обучением GNN: память, процессор, видеокарта, диск, PyTorch."""
import ctypes
import os
import platform
import shutil
import subprocess
import sys
from pathlib import Path

out = []


def log(*a):
    s = " ".join(str(x) for x in a)
    print(s, flush=True)
    out.append(s)


log("Python:", sys.version.split()[0], "|", platform.platform())
log("Ядер CPU (логических):", os.cpu_count())
try:
    class MEM(ctypes.Structure):
        _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                    ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                    ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                    ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                    ("sullAvailExtendedVirtual", ctypes.c_ulonglong)]
    m = MEM()
    m.dwLength = ctypes.sizeof(MEM)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    log(f"ОЗУ: всего {m.ullTotalPhys / 2**30:.1f} ГБ, свободно {m.ullAvailPhys / 2**30:.1f} ГБ")
except Exception as e:
    log("ОЗУ: не удалось определить:", e)
du = shutil.disk_usage(Path.home())
log(f"Диск (домашняя папка): свободно {du.free / 2**30:.1f} ГБ")
try:
    r = subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version", "--format=csv,noheader"],
                       capture_output=True, text=True, timeout=20)
    log("NVIDIA GPU:", r.stdout.strip() if r.returncode == 0 and r.stdout.strip() else "нет")
except Exception:
    log("NVIDIA GPU: нет (nvidia-smi не найден)")
try:
    import torch
    log("PyTorch:", torch.__version__, "| CUDA доступна:", torch.cuda.is_available(),
        "|", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
    log("Потоков torch:", torch.get_num_threads())
except Exception as e:
    log("PyTorch: не установлен (", e, ")")
(Path(__file__).resolve().parent / "results_05a_env.txt").write_text("\n".join(out), encoding="utf-8")

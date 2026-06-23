# RAMガード — モデルロードでPCをフリーズさせないための物理的な歯止め(2026-06-22)
# 教訓: gemma-2-2b(F32 10.4GB)をTransformerLensでロード→変換で膨れRAM98%。
# パラメータ数ではなく「チェックポイントのバイト数 × 変換オーバーヘッド」が危険量。
#
# 使い方(モデルロードの直前に):
#   from ram_guard import start_watchdog, assert_safe_to_load
#   start_watchdog(80)                 # 80%超えたら自己終了(フリーズ前に止める)
#   assert_safe_to_load("google/gemma-2-2b")  # 推定peak > free なら中止
import ctypes
import os
import sys
import threading
import time


class _MEMSTAT(ctypes.Structure):
    _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong),
                ("ullTotalPhys", ctypes.c_ulonglong), ("ullAvailPhys", ctypes.c_ulonglong),
                ("ullTotalPageFile", ctypes.c_ulonglong), ("ullAvailPageFile", ctypes.c_ulonglong),
                ("ullTotalVirtual", ctypes.c_ulonglong), ("ullAvailVirtual", ctypes.c_ulonglong),
                ("ullAvailExtendedVirtual", ctypes.c_ulonglong)]


def _mem():
    m = _MEMSTAT()
    m.dwLength = ctypes.sizeof(m)
    ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(m))
    return m


GB = 1024 ** 3


def start_watchdog(threshold_pct: int = 80, poll_sec: float = 0.3) -> None:
    """RAM使用率がthreshold_pctを超えたら即os._exit。フリーズ前にプロセスを殺す物理的歯止め。"""
    def _watch():
        while True:
            load = _mem().dwMemoryLoad
            if load >= threshold_pct:
                sys.stderr.write(
                    f"\n[RAM-GUARD] メモリ {load}% >= {threshold_pct}% — フリーズ防止のため強制終了\n")
                sys.stderr.flush()
                os._exit(99)
            time.sleep(poll_sec)
    threading.Thread(target=_watch, daemon=True).start()
    m = _mem()
    print(f"[RAM-GUARD] watchdog起動(閾値{threshold_pct}%)。現在 used={m.dwMemoryLoad}% "
          f"free={m.ullAvailPhys/GB:.1f}GB/{m.ullTotalPhys/GB:.1f}GB")


def _checkpoint_bytes(model_name: str) -> int:
    """ダウンロードせずにチェックポイント(safetensors/bin)合計バイトを得る。0=不明。"""
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(model_name, files_metadata=True)
        tot = sum((s.size or 0) for s in info.siblings
                  if s.rfilename.endswith((".safetensors", ".bin")))
        return tot
    except Exception as e:  # noqa: BLE001
        print(f"[RAM-GUARD] サイズ照会失敗({e})。watchdogのみで進む")
        return 0


def assert_safe_to_load(model_name: str, overhead: float = 2.5) -> None:
    """推定ピークRAM(=checkpoint×overhead) > 空きRAM なら中止。
    overhead=2.5 はTransformerLensのHFロード+変換でチェックポイントの約2.5倍まで膨れる実測根拠
    (gemma-2-2b F32 10.4GB→OOM, Qwen3-4B bf16 8GB→OK の境界に整合)。"""
    free = _mem().ullAvailPhys
    ck = _checkpoint_bytes(model_name)
    if ck == 0:
        print(f"[RAM-GUARD] {model_name}: サイズ不明のためwatchdog任せ。慎重に。")
        return
    peak = ck * overhead
    print(f"[RAM-GUARD] {model_name}: checkpoint={ck/GB:.1f}GB 推定peak(×{overhead})={peak/GB:.1f}GB "
          f"空き={free/GB:.1f}GB")
    if peak > free:
        sys.exit(f"[RAM-GUARD] 中止: 推定peak {peak/GB:.1f}GB > 空き {free/GB:.1f}GB。"
                 f"フリーズ防止のためロードしない。\n"
                 f"  → 対策: より小さいモデル / bf16保存リポ / 量子化 / クラウドを使う。"
                 f"gemma-2-2b(F32)はこのマシン(32GB)では不可。")


if __name__ == "__main__":
    # 単体テスト: 現在のメモリと、引数モデルの安全判定
    start_watchdog(80)
    if len(sys.argv) > 1:
        assert_safe_to_load(sys.argv[1])

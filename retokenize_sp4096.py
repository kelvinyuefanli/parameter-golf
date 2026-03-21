"""Retokenize FineWeb data from SP1024 to SP4096.
Run this on RunPod before training with SP4096.
Takes ~5 min on 16 vCPU. Requires SP1024 data already downloaded.
"""
import numpy as np
import sentencepiece as spm
from pathlib import Path
import time

SP1024_DIR = Path("data/datasets/fineweb10B_sp1024")
SP4096_DIR = Path("data/datasets/fineweb10B_sp4096")
SP1024_MODEL = "data/tokenizers/fineweb_1024_bpe.model"
SP4096_MODEL = "data/tokenizers/fineweb_4096_bpe.model"
CHUNK_SIZE = 100_000
MAGIC = 20240520
VERSION = 1

def retokenize_shard(sp1, sp4, src_file, dst_file):
    data = np.fromfile(src_file, dtype="<u2", offset=1024)
    all_ids = []
    for start in range(0, len(data), CHUNK_SIZE):
        chunk = data[start:start+CHUNK_SIZE].tolist()
        text = sp1.decode(chunk)
        all_ids.extend(sp4.encode(text))
    ids = np.array(all_ids, dtype=np.uint16)
    header = np.zeros(256, dtype=np.int32)
    header[0], header[1], header[2] = MAGIC, VERSION, len(ids)
    with open(dst_file, "wb") as f:
        f.write(header.tobytes())
        f.write(ids.tobytes())
    return len(data), len(ids)

def main():
    sp1 = spm.SentencePieceProcessor(model_file=SP1024_MODEL)
    sp4 = spm.SentencePieceProcessor(model_file=SP4096_MODEL)
    SP4096_DIR.mkdir(parents=True, exist_ok=True)

    for src in sorted(SP1024_DIR.glob("fineweb_*.bin")):
        dst = SP4096_DIR / src.name
        if dst.exists():
            print(f"  skip {src.name} (exists)")
            continue
        t0 = time.time()
        n1, n4 = retokenize_shard(sp1, sp4, src, dst)
        print(f"  {src.name}: {n1:,} → {n4:,} tokens ({n4/n1*100:.1f}%) in {time.time()-t0:.0f}s")

    print("Done!")

if __name__ == "__main__":
    main()

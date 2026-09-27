from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from gguf import GGMLQuantizationType, GGUFReader


PQ2_BLOCK_SIZE = 128
PQ2_BLOCK_BYTES = 34
PQ2_QUANT_BYTES = 32
BLOCKS_PER_CHUNK = 1 << 18


def validate_model(path: Path) -> tuple[int, int]:
    reader = GGUFReader(path, "r")
    tensor_count = 0
    quant_count = 0

    for tensor in reader.tensors:
        if tensor.tensor_type != GGMLQuantizationType.PQ2_0:
            continue

        if int(tensor.shape[0]) % PQ2_BLOCK_SIZE != 0:
            raise ValueError(f"{path}: {tensor.name}: row width is not a multiple of {PQ2_BLOCK_SIZE}")

        raw = tensor.data.reshape(-1)
        if raw.size % PQ2_BLOCK_BYTES != 0:
            raise ValueError(f"{path}: {tensor.name}: invalid PQ2_0 block byte count")

        blocks = raw.reshape(-1, PQ2_BLOCK_BYTES)
        tensor_count += 1
        quant_count += int(blocks.shape[0]) * PQ2_BLOCK_SIZE

        for block_start in range(0, blocks.shape[0], BLOCKS_PER_CHUNK):
            block_end = min(block_start + BLOCKS_PER_CHUNK, blocks.shape[0])
            quants = blocks[block_start:block_end, 2:]
            matches = np.flatnonzero((quants & (quants >> 1) & 0x55) != 0)
            if matches.size == 0:
                continue

            block_offset, byte_offset = divmod(int(matches[0]), PQ2_QUANT_BYTES)
            byte = int(quants[block_offset, byte_offset])
            for shift in (0, 2, 4, 6):
                if ((byte >> shift) & 0x03) == 0x03:
                    element = ((block_start + block_offset) * PQ2_BLOCK_SIZE + byte_offset * 4 + shift // 2)
                    raise ValueError(f"{path}: {tensor.name}: q=3 at element {element}")

    return tensor_count, quant_count


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Verify that every PQ2_0 code in GGUF files is in {0, 1, 2} for the Bonsai ternary Vulkan path."
    )
    parser.add_argument("models", nargs="+", type=Path, help="GGUF model file or all shards of a split model")
    args = parser.parse_args()

    total_tensors = 0
    total_quants = 0
    try:
        for path in args.models:
            tensor_count, quant_count = validate_model(path)
            total_tensors += tensor_count
            total_quants += quant_count
            print(f"{path}: validated {tensor_count} PQ2_0 tensors ({quant_count} codes)")
    except (OSError, ValueError, KeyError) as exc:
        print(f"PQ2_0 Bonsai validation failed: {exc}", file=sys.stderr)
        return 1

    if total_tensors == 0:
        print("PQ2_0 Bonsai validation failed: no PQ2_0 tensors found", file=sys.stderr)
        return 1

    print(f"Validated {total_tensors} PQ2_0 tensors ({total_quants} codes); no q=3 codes found.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

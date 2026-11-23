"""Assemble the locked Pair05 foreground files in a fresh inference directory."""

import argparse
import hashlib
from pathlib import Path
import sys

from check_inputs import DEPTHS, EXO, PRIOR, sha256


PATCHED_CORE = Path("core/finetune/models/wan_i2v/custom_transformer.py")
PATCHED = {
    PATCHED_CORE: "5c5206a1acef19101ae78eb1f877c78efb964221f1299ad474cff11a49a4c15c",
    Path("infer.py"): "b5f82acb510cae7c1a98077baf35e3414612309aed72caf3082d728c24669e00",
    Path("pair05_gga_routes.py"): "ac75e34e3ac2fc7c3c83d854a1be2e7b10a47b6b5c46263a26a9472d3083f090",
    Path("aria_gga.py"): "e23b80c36a1c828ee4d570694ff1cecbe33f0cdb5b481534056fd14ea19eb7bf",
}
FROZEN = {
    Path("meta.json"): "c2422ed6f6ecb0c150b8be612bc7c927cc39f5ec70cdd5a0593d8811d8d5846c",
    Path("validated-inputs.json"): "9d5623b47a85468b076c3bb68edb8ffef4d8b236a95ff32c02712df37fdc06f9",
    Path("plan/pair05-gga-routes.npz"): "3cd25a102a6ecf4006add800b293dcf40b82b296392ff652aeea308fcd0f501e",
}


def verified_payload(bundle, official):
    source = bundle / "staged"
    unchanged = {
        path.relative_to(official): sha256(path)
        for path in (official / "core").rglob("*.py")
    }
    if len(unchanged) != 30 or PATCHED_CORE not in unchanged:
        raise ValueError("Expected the 30 Python files from the stock EgoX core.")
    expected = unchanged | PATCHED
    actual = {path.relative_to(source) for path in source.rglob("*") if path.is_file() or path.is_symlink()}
    if actual != set(expected):
        raise ValueError("The staged Python payload has missing or unexpected files.")
    payload = {}
    for root, checksums in ((source, expected), (bundle, FROZEN)):
        for relative, digest in checksums.items():
            path = root / relative
            if path.is_symlink():
                raise ValueError(f"Expected a regular packaged file: {relative}")
            contents = path.read_bytes()
            if hashlib.sha256(contents).hexdigest() != digest:
                raise ValueError(f"Unexpected staged SHA-256: {relative}")
            payload[relative] = contents
    return payload


def assemble(bundle, official, media, stage):
    stage, official = stage.resolve(), official.resolve(strict=True)
    if stage == official or official in stage.parents:
        raise ValueError("Stage the foreground code outside the official inference directory.")
    payload = verified_payload(bundle, official)
    stage.mkdir(exist_ok=False)
    for relative, contents in payload.items():
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("xb") as stream:
            stream.write(contents)
    depths = sorted((media / DEPTHS).glob("*.npy"))
    for relative in [EXO, PRIOR] + [path.relative_to(media) for path in depths]:
        target = stage / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.symlink_to((media / relative).resolve(strict=True))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--media-root", type=Path, required=True)
    parser.add_argument("--stage", type=Path, required=True)
    args = parser.parse_args()
    try:
        assemble(Path(__file__).resolve().parent, args.official, args.media_root, args.stage)
    except (OSError, ValueError) as error:
        print(f"PAIR05_FOREGROUND_STAGE_FAILED {error}", file=sys.stderr)
        return 1
    print("PAIR05_FOREGROUND_STAGE_OK", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

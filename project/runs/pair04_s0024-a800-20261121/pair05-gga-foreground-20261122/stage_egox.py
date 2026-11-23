"""Materialize a Pair05-only optional attention hook from the pinned EgoX checkout."""

import hashlib
import json
from pathlib import Path
import shutil
import sys

HERE = Path(__file__).resolve().parent
UPSTREAM = HERE.parents[2] / "vendor/EgoX"
TRANSFORMER = Path("core/finetune/models/wan_i2v/custom_transformer.py")
PINNED_TRANSFORMER = "ffe4f15c544cdc4869495b720a0a79a2989c758ffe7e3a332f0f2cc3b4fca079"
PINNED_INFER = "99199fac22858821d356f0ec3b8577495290dd579a8d80368de1fca68d559bed"
OPEN_CV = "            normalized_points = cv2.undistortPoints(pixel_coords_cv, K, D, R=np.eye(3), P=np.eye(3))\n"
ARIA = """            from aria_gga import undistort_aria_for_gga
            normalized_points = undistort_aria_for_gga(pixel_coords_cv, K, D)
"""


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_pinned(relative, expected):
    path = UPSTREAM / relative
    if digest(path) != expected:
        raise ValueError(f"Upstream seam changed: {relative}")
    return path.read_text()


def replace_once(source, anchor, replacement):
    if source.count(anchor) != 1:
        raise ValueError(f"Unexpected occurrence count for {anchor.strip()!r}.")
    return source.replace(anchor, replacement, 1)


def stage(output):
    transformer = read_pinned(TRANSFORMER, PINNED_TRANSFORMER)
    infer = read_pinned("infer.py", PINNED_INFER)
    transformer = replace_once(transformer, "        if not hasattr(F, \"scaled_dot_product_attention\"):\n",
                               "        self._pair05_foreground = None\n"
                               "        if not hasattr(F, \"scaled_dot_product_attention\"):\n")
    native_attention = "        if cos_sim is not None and not do_kv_cache:\n"
    transformer = replace_once(transformer, native_attention, """        if self._pair05_foreground is not None and (cos_sim is None or do_kv_cache):
            raise RuntimeError("Pair05 foreground routing requires GGA and no KV cache in every self-attention call.")

""" + native_attention)
    transformer = replace_once(transformer, """            hidden_states_ego = F.scaled_dot_product_attention(
                query[:, :, attention_mask_GGA==0, :],
""", """            ego_query = query[:, :, attention_mask_GGA==0, :]
            hidden_states_ego = F.scaled_dot_product_attention(
                ego_query,
""")
    seam = "            #! Combine exo and ego\n"
    transformer = replace_once(transformer, seam, """            if self._pair05_foreground is not None:
                hidden_states_ego = self._pair05_foreground.apply(
                    hidden_states_ego, ego_query,
                    key, value, attn_mask_ego, attention_mask_GGA == 1,
                )

""" + seam)
    infer = replace_once(infer, OPEN_CV, ARIA)
    infer = replace_once(infer, "    import shutil\n", """    foreground_routes = None
    if args.foreground_gga_plan is not None:
        if args.idx not in {-1, 0} or args.start_idx != -1:
            raise ValueError("The Pair05 foreground plan requires running the single reviewed row.")
        from pair05_gga_routes import load_foreground_routes
        foreground_routes = load_foreground_routes(
            args.foreground_gga_plan, meta_data_file, meta_data, take_names,
            args.use_GGA, args.do_kv_cache, args.cos_sim_scaling_factor,
        )

    import shutil
""")
    infer = replace_once(infer, '    pipe.to("cuda")\n', """    if foreground_routes is not None:
        from pair05_gga_routes import attach_foreground_routes
        attach_foreground_routes(pipe.transformer, foreground_routes)

    pipe.to("cuda")
""")
    native_start = "            point_map = torch.nn.functional.interpolate(point_map, size=(exo_H, exo_W),"
    native_end = "            cam_rays = torch.rot90(cam_rays, k=-1, dims=[1, 2])"
    native = infer[infer.index(native_start):infer.index(native_end)]
    origin_start = native.index("            ego_extrinsic_c2w = ")
    origin_end = native.index("            if point_map.size(0)")
    origins = native[origin_start:origin_end]
    native_without_origins = native[:origin_start] + native[origin_end:]
    native_indented = "".join("    " + line if line.strip() else "\n"
                              for line in native_without_origins.splitlines(keepends=True))
    infer = replace_once(infer, native, origins + """            if foreground_routes is None:
""" + native_indented + """            else:
                point_vecs_per_frame = foreground_routes.geometry_directions(point_map, cam_origins)
                point_vecs = torch.stack([point_vecs_per_frame[j, j] for j in range(F)])

""")
    infer = replace_once(infer, '\n\nif __name__ == "__main__":\n', """
    if foreground_routes is not None:
        if foreground_routes.attention_calls == 0:
            raise RuntimeError("Pair05 finished without using the configured foreground GGA route.")
        print(f"Pair05 foreground GGA attention calls: {foreground_routes.attention_calls}")


if __name__ == "__main__":
""")
    infer = replace_once(infer, "    args = parser.parse_args()\n", """    parser.add_argument("--foreground_gga_plan", default=None, help="Optional verified Pair05 GGA plan")
    args = parser.parse_args()
""")
    compile(transformer, str(TRANSFORMER), "exec")
    compile(infer, "infer.py", "exec")
    if output.resolve() == UPSTREAM.resolve() or UPSTREAM.resolve() in output.resolve().parents:
        raise ValueError("Stage the hook outside the official EgoX checkout.")
    output.mkdir(parents=True)
    shutil.copytree(UPSTREAM / "core", output / "core", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    (output / TRANSFORMER).write_text(transformer)
    (output / "infer.py").write_text(infer)
    shutil.copyfile(HERE / "pair05_gga_routes.py", output / "pair05_gga_routes.py")
    shutil.copyfile(HERE.parent / "pair01-gga-ab/aria_gga.py", output / "aria_gga.py")
    return {str(name): digest(output / name) for name in (TRANSFORMER, "infer.py", "pair05_gga_routes.py", "aria_gga.py")}


if __name__ == "__main__":
    if len(sys.argv) != 2:
        raise SystemExit("Usage: stage_egox.py NEW_EGOX_DIRECTORY")
    print(json.dumps(stage(Path(sys.argv[1])), indent=2))

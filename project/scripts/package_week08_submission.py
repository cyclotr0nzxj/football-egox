#!/usr/bin/env python3
"""Prepare or finalize a portable local Week 08 submission repository.

--prepare copies only the four input JPEGs, capture figure and selected evidence.
--finalize additionally copies the final code/results/report, initializes local
Git without an identity or commit, and verifies 10 named trimmed 4K sources plus 3 work videos.
"""
from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import io
import json
import os
from pathlib import Path
import re
import subprocess
from typing import Any
import zipfile

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "submission/week08_harris_sift"
SOURCE_REPORT = ROOT / "output/week08_harris_sift"
FRAME_SOURCE = ROOT / "output/week08/frames/scene04"
CAPTURE_SOURCE = ROOT / "output/week08/report_assets/capture_setup.jpg"
DYNAMIC_SOURCE = SOURCE_REPORT / "dynamic"
PAPER_FIGURE_SOURCE = SOURCE_REPORT / "paper_figures"
VIDEO_PATHS = {
    "exo.mp4": "data/video_pair04/exo.mp4",
    "ego.mp4": "data/video_pair04/ego.mp4",
    "动态特征检测_双视角对照.mp4": "videos/动态特征检测_双视角对照.mp4",
}
STATE = TARGET / ".package_state.json"
ARCHIVE = ROOT / "dist/第8周_Harris与SIFT_提交包.zip"
MAX_ARCHIVE_BYTES = 2 * 1024 ** 3
SOURCE_VIDEO_PREFIX = "data/source_videos_4k"
SYNC_MANIFEST = ROOT / "records/capture_sync_20261019/sync_manifest.json"
TRIM_MANIFEST = ROOT / "records/user_ego_trim_20261019/trim_manifest.json"


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def stream_digest(handle: Any) -> str:
    h = hashlib.sha256()
    for block in iter(lambda: handle.read(4 * 1024 * 1024), b""):
        h.update(block)
    return h.hexdigest()


def sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return stream_digest(handle)


def encode_json(data: Any) -> bytes:
    return (json.dumps(data, ensure_ascii=False, indent=2) + "\n").encode("utf-8")


def project_path(path: Path) -> str:
    return path.resolve().relative_to(ROOT).as_posix()


class Package:
    def __init__(self) -> None:
        if STATE.exists():
            self.state = json.loads(STATE.read_text(encoding="utf-8"))
        else:
            if TARGET.exists() and any(TARGET.iterdir()):
                raise FileExistsError(f"Unmanaged non-empty submission directory; inspect before use: {TARGET}")
            self.state = {"schema_version": 1, "managed": {}, "sources": {}}
        self.path_map: dict[str, str] = {}

    def put(self, relative: str, data: bytes, source: Path | None = None, note: str = "") -> None:
        target = TARGET / relative
        if target.is_symlink() or not target.resolve().is_relative_to(TARGET.resolve()):
            raise ValueError(f"Unsafe submission path: {relative}")
        desired = digest(data)
        if target.exists():
            actual = sha256(target)
            previous = self.state["managed"].get(relative)
            if actual != desired and actual != previous:
                raise FileExistsError(f"Refusing to replace a user-modified/unmanaged file: {target}")
        if not target.exists() or sha256(target) != desired:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".package-partial")
            partial.write_bytes(data)
            os.replace(partial, target)
        self.state["managed"][relative] = desired
        if source is not None:
            self.state["sources"][relative] = {
                "source_project_relative_path": project_path(source),
                "source_sha256": sha256(source),
                "packaged_sha256": desired,
                "content_rewritten": desired != sha256(source),
                "note": note,
            }

    def copy(self, source: Path, relative: str) -> None:
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Expected regular source file: {source}")
        self.put(relative, source.read_bytes(), source)

    def copy_stream(self, source: Path, relative: str) -> dict[str, Any]:
        """Copy a large source without buffering it, checking source before/after."""
        if not source.is_file() or source.is_symlink():
            raise FileNotFoundError(f"Expected regular source file: {source}")
        target = TARGET / relative
        if target.is_symlink() or not target.resolve().is_relative_to(TARGET.resolve()):
            raise ValueError(f"Unsafe submission path: {relative}")
        before = sha256(source)
        source_bytes = source.stat().st_size
        actual = sha256(target) if target.exists() else None
        previous = self.state["managed"].get(relative)
        if actual is not None and actual not in (before, previous):
            raise FileExistsError(f"Refusing to replace a user-modified/unmanaged file: {target}")
        if actual != before:
            target.parent.mkdir(parents=True, exist_ok=True)
            partial = target.with_name(target.name + ".package-partial")
            copied = hashlib.sha256()
            try:
                with source.open("rb") as src, partial.open("wb") as dst:
                    for block in iter(lambda: src.read(4 * 1024 * 1024), b""):
                        dst.write(block)
                        copied.update(block)
                if copied.hexdigest() != before or partial.stat().st_size != source_bytes:
                    raise ValueError(f"Source changed while streaming copy: {source}")
                os.replace(partial, target)
            finally:
                partial.unlink(missing_ok=True)
        after = sha256(source)
        packaged = sha256(target)
        if before != after or packaged != before or source.stat().st_size != source_bytes:
            raise ValueError(f"Source or packaged bytes changed during copy verification: {source}")
        self.state["managed"][relative] = packaged
        self.state["sources"][relative] = {
            "source_project_relative_path": project_path(source),
            "source_sha256": before, "source_sha256_before": before,
            "source_sha256_after": after, "packaged_sha256": packaged,
            "content_rewritten": False, "source_unchanged_after_copy": True,
            "note": "Streamed byte-for-byte; already synchronized/trimmed 4K source, not an unprocessed camera raw recording."
        }
        return {"source_sha256_before": before, "source_sha256_after": after,
                "packaged_sha256": packaged, "source_unchanged_after_copy": True,
                "bytes": source_bytes}

    def save_state(self) -> None:
        TARGET.mkdir(parents=True, exist_ok=True)
        STATE.write_bytes(encode_json(self.state))

    def retire(self, relative: str) -> None:
        """Remove only a previous tool-managed obsolete package copy."""
        path = TARGET / relative
        if path.exists():
            if sha256(path) != self.state["managed"].get(relative):
                raise FileExistsError(f"Refusing to retire a user-modified file: {path}")
            path.unlink()
        self.state["managed"].pop(relative, None)
        self.state["sources"].pop(relative, None)

    def rewrite_paths(self, value: Any) -> Any:
        if isinstance(value, dict):
            return {key: self.rewrite_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.rewrite_paths(item) for item in value]
        if not isinstance(value, str):
            return value
        result = value
        # Longest paths first avoids a directory prefix consuming a file map.
        for source, target in sorted(self.path_map.items(), key=lambda item: len(item[0]), reverse=True):
            if result == source or result.startswith(source + "/"):
                return target + result[len(source):]
        if result.startswith(str(ROOT) + "/"):
            return "source_project_reference/" + result[len(str(ROOT)) + 1:]
        return result

    def copy_json(self, source: Path, relative: str) -> None:
        content = json.loads(source.read_text(encoding="utf-8"))
        self.put(relative, encode_json(self.rewrite_paths(content)), source,
                 "Only path strings are rewritten for portability; numeric results are unchanged.")


def historical_markdown(source: Path) -> bytes:
    text = source.read_text(encoding="utf-8")
    prefix = str(ROOT) + "/"
    pattern = re.compile(r"\[([^\]]+)\]\(" + re.escape(prefix) + r"([^\)]+)\)")
    text = pattern.sub(lambda match: match[1] + "（源项目相对路径：`" + match[2] + "`）", text)
    text = text.replace(prefix, "")
    header = ("# 历史溯源文件副本\n\n"
              "以下记录来自源项目；历史未裁剪路径与旧脚本不是本包运行入口。最终裁剪后视频另见data/source_videos_4k。"
              "本包可运行入口见根目录 README。原有数值与结论保留，仅将绝对本机链接转为源项目相对路径文字。\n\n")
    return (header + text).encode("utf-8")


def readme() -> bytes:
    text = """# 第8周 Harris 与 SIFT 本地提交仓库

这是**本地 Git 仓库**，ZIP 不含 `.git/`；解压后即可按下面命令运行。

## 内容与采集流程

R8/Nano 采集 → 音画同步 → 裁剪选帧 → 受控旋转/缩放/模糊样本生成 → 灰度与 Harris/SIFT 检测 → 图与指标 → 实验报告 PDF。

`data/frames/scene04/{exo,ego}/0050.jpg` 和 `0200.jpg` 共4张960×540 JPEG用于静态受控实验。scene04指**配对编号04，现场场景2**。原视频采集为3840×2160、25fps；F50/F200对应用户最终裁剪片段的2.00/8.00秒，精确映射与SHA256见 `data/manifest.json`、`data/frame_mapping.json`。

`data/capture_setup.jpg` 是现场场景2的两张机位记录拼图。新增动态证据使用 `data/video_pair04/exo.mp4` 与 `ego.mp4` 两路960×540工作副本，覆盖约12.44秒。`evidence/` 保存采集、同步、裁剪与动态运行证据。`data/source_videos_4k/trimmed_pairs/`另外附上5组最终裁剪后的4K EXO/EGO素材，共10条；它们已同步并按用户区间裁剪，打包时不转码。本包不含模型。

- `scripts/`：Harris/SIFT实验、独立审计、论文图绘制、报告组装与PDF构建脚本。
- `results/`：正式运行的参数、48条逐图/条件/算法记录、坐标、受控变换、掩膜、图与审计。
- `report/`：以采集、处理过程和完整结果为主的正式实验报告，包含少量摘要、关键词等论文元素；另有可编辑Markdown、同源JSON与正文附图。`report/paper_figures/`保留三张科研图的PNG/PDF/SVG、图源数据和作图QA，作为补充材料。
- `data/source_videos_4k/`：裁剪后的5组4K双视角源素材，共10条；manifest.csv/JSON记录原相对路径、编码、时长、可见帧数、大小与SHA256。
- `videos/动态特征检测_双视角对照.mp4`：直接观看两视角逐帧Harris/SIFT检测；加上两条960×540工作副本，包内共13个视频。MP4使用ZIP_STORED，保持原有字节。
- `evidence/history/`：上一版静态提交记录及历史6页论文稿的审阅/复现记录；其旧脚本与报告哈希不代表本轮实验报告。当前验收见 `evidence/delivery_validation.json`、`report_revision_review.json`和`report_portable_reproduction.json`；修订说明见 `evidence/revision_notes.md`。原48组结果与动态视频保持不变。
- `manifest.json`、`SHA256SUMS`：打包路径映射、源/提交文件哈希及载荷清单。SHA256SUMS包含manifest本身，自身不做递归哈希。

观看 `videos/动态特征检测_双视角对照.mp4`。从包内工作副本复现动态演示到空目录：

```sh
python scripts/build_feature_video.py --exo data/video_pair04/exo.mp4 --ego data/video_pair04/ego.mp4 --output dynamic_reproduced
python scripts/audit_feature_video.py --dynamic evidence/dynamic --static-results results --output dynamic_audit_reproduced.json
```

动态审计检查时间戳、文件与计数，并用同一检测函数抽样复算；不将其称为独立算法验证。

## 科学边界

本期用同一短片的两个时刻、两个视角，共4图；旋转、缩放和模糊是**受控样本生成**，不是新拍摄视角或EgoX生成。高位置重复率也可能来自较小分母：一个旋转案例的位置重复率r为97.1%，源点保留比例却只有17.3%，应结合点数与对应数理解。实验不评价描述子、语义准确率或三维重建，4图也不能支持泛化结论。变换JPEG仅供展示，精确复现必须从4张原始工作图在内存重新生成。

动态视频是每帧重新检测和绘制特征，**不分配持续点ID，不是特征跟踪**；人物与球的运动不能据此称为已跟踪。动态展示不改变静态实验的48条记录与24份坐标。

本期仅在本地CPU运行。第14周的单目深度辅助三维重建与自采EgoX尚未运行，不属于本作业成果。

## 环境与运行

在解压后的仓库根目录执行，使用Python3.12。`requirements.txt` 只安装Python包；动态视频另需系统 **FFmpeg/ffprobe**，并保证二者在PATH中可执行。无需模型依赖：

```sh
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

复现到新的空目录，避免覆盖正式结果：

```sh
python scripts/course_feature_detection.py \\
  --frames-root data/frames/scene04 \\
  --output results_reproduced
python scripts/audit_feature_results.py \\
  --results results_reproduced \\
  --frames-root data/frames/scene04 \\
  --output independent_audit_reproduced.json
python scripts/assemble_feature_report.py \\
  --results results_reproduced \\
  --output report_reproduced/report_data.json \\
  --capture-evidence data/capture_setup.jpg \\
  --paper-figures report/paper_figures \\
  --dynamic evidence/dynamic
python scripts/build_week08_pdf.py --check-fonts
python scripts/build_week08_pdf.py \\
  --data report_reproduced/report_data.json \\
  --output report_reproduced/第8周_Harris与SIFT实验报告.pdf \\
  --markdown report_reproduced/实验报告.md
```

科研图作为补充材料保留；重建报告无需先重绘科研图。若需单独复绘，运行：

```sh
python scripts/plot_week08_paper_figures.py \\
  --results results \\
  --frames-root data/frames/scene04 \\
  --output paper_figures_reproduced \\
  --qa-tools report/paper_figures/qa_tools
```

PDF需要可嵌入的中文TrueType字体；macOS默认查找Arial Unicode/STHeiti，其他系统可用构建器的 `--font` 指定本机中文字体文件。算法图使用可用的系统英文字体。无需访问源项目的绝对目录或云服务器；数据与视频按包内清单提供。

坐标与重复率应可复核；CPU单次耗时是描述性记录，会随机器变化。新PDF生成后应再逐页检查排版。
"""
    return text.encode("utf-8")


def prepare(package: Package) -> None:
    source_mapping_path = FRAME_SOURCE / "frame_mapping.json"
    package.path_map[str(FRAME_SOURCE)] = "data/frames/scene04"
    package.path_map[project_path(FRAME_SOURCE)] = "data/frames/scene04"
    mapping = json.loads(source_mapping_path.read_text(encoding="utf-8"))
    selected = [item for item in mapping["frames"] if item["visible_frame"] in (50, 200)]
    if len(selected) != 2 or mapping["pair_id"] != "04":
        raise ValueError("Expected exact frame50/frame200 mapping for pair04")
    image_rows = []
    for view in ("exo", "ego"):
        for index in (50, 200):
            source = FRAME_SOURCE / view / f"{index:04d}.jpg"
            with Image.open(source) as image:
                image.verify()
            with Image.open(source) as image:
                if image.size != (960, 540):
                    raise ValueError(f"Unexpected source size: {source}")
            target = f"data/frames/scene04/{view}/{index:04d}.jpg"
            package.copy(source, target)
            package.path_map[str(source)] = target
            package.path_map[project_path(source)] = target
            match = next(item for item in selected if item["visible_frame"] == index)
            image_rows.append({"path": target, "source_project_relative_path": project_path(source),
                               "sha256": sha256(source), "width": 960, "height": 540,
                               "view": view, "visible_frame": index,
                               "trimmed_clip_pts_seconds": match["source_pts_seconds"],
                               "base_aligned_frame": match["base_aligned_frame"],
                               "historical_raw_exo_frame": match["historical_raw_exo_frame"],
                               "historical_raw_exo_frame_is_history_only": True})
    package.copy(CAPTURE_SOURCE, "data/capture_setup.jpg")
    package.path_map[str(CAPTURE_SOURCE)] = "data/capture_setup.jpg"
    package.path_map[project_path(CAPTURE_SOURCE)] = "data/capture_setup.jpg"
    portable_mapping = dict(mapping)
    portable_mapping["selected_visible_frames"] = [50, 200]
    portable_mapping["frames"] = selected
    portable_mapping["trim_manifest"] = "evidence/trim_manifest.json"
    portable_mapping["source_mapping_project_relative_path"] = project_path(source_mapping_path)
    portable_mapping["package_root_relative_paths"] = True
    portable_mapping["source_videos_included_in_submission"] = True
    portable_mapping["packaged_source_video_manifest"] = "data/source_videos_4k/manifest.json"
    for frame in portable_mapping["frames"]:
        frame["images"] = {view: f"data/frames/scene04/{view}/{frame['visible_frame']:04d}.jpg" for view in ("exo", "ego")}
    package.put("data/frame_mapping.json", encode_json(portable_mapping), source_mapping_path,
                "Subset to frames50/200, images use submission-root relative paths; numeric mappings unchanged.")
    package.put("data/manifest.json", encode_json({
        "schema_version": 1, "pair_id": "04", "field_scene_id": "2",
        "scene04_meaning": "paired clip04, field scene2; not field scene4",
        "capture_fps": 25, "source_video_size": [3840, 2160], "working_size": [960, 540],
        "source_videos_included": True,
        "packaged_source_video_manifest": "data/source_videos_4k/manifest.json",
        "source_video_provenance": mapping["source_videos"],
        "synchronization_uncertainty_frames": 1, "images": image_rows,
        "capture_evidence": {"path": "data/capture_setup.jpg", "sha256": sha256(CAPTURE_SOURCE),
                             "source_project_relative_path": project_path(CAPTURE_SOURCE)},
        "note": "Four JPEGs reproduce detection experiments. Five final trimmed 4K pairs are included under data/source_videos_4k; historical uncut-frame indices are provenance only, not a claim of recovering uncut camera recordings."
    }))
    evidence = {
        "records/course_validation_20261019/capture_metadata.json": "evidence/capture_metadata.json",
        "records/course_validation_20261019/capture_metadata.md": "evidence/capture_metadata.md",
        "records/user_ego_trim_20261019/trim_manifest.json": "evidence/trim_manifest.json",
        "records/capture_sync_20261019/sync_manifest.json": "evidence/sync_manifest.json",
        "records/user_ego_trim_20261019/packet_match_summary.json": "evidence/packet_match_summary.json",
    }
    for source, target in evidence.items():
        package.copy(ROOT / source, target)
    package.copy(source_mapping_path, "evidence/frame_mapping_source.json")
    for source, target in ((ROOT / "验证视频/同步记录_20261019.md", "evidence/synchronization_record.md"),
                           (ROOT / "验证视频/裁剪后验证exo/README.md", "evidence/trim_record.md")):
        package.put(target, historical_markdown(source), source,
                    "Historical document; local absolute links converted to source-project relative text.")
    package.put("README.md", readme())
    package.copy(SOURCE_REPORT / "requirements.txt", "requirements.txt")
    package.put(".gitignore", b".DS_Store\n__pycache__/\n*.py[cod]\n.venv*/\nresults_reproduced*/\nreport_reproduced*/\n.package_state.json\n*.package-partial\n")
    package.save_state()


def image_target(package: Package, source: Path) -> str:
    source = source.resolve()
    if str(source) in package.path_map:
        return package.path_map[str(source)]
    if source.is_relative_to(PAPER_FIGURE_SOURCE):
        return "report/paper_figures/" + source.relative_to(PAPER_FIGURE_SOURCE).as_posix()
    if source.is_relative_to(SOURCE_REPORT / "results"):
        return "results/" + source.relative_to(SOURCE_REPORT / "results").as_posix()
    if source.is_relative_to(SOURCE_REPORT):
        relative = source.relative_to(SOURCE_REPORT)
        if relative.parts[0] in ("assets", "report_assets"):
            relative = Path(*relative.parts[1:])
        return "report/assets/" + relative.as_posix()
    raise ValueError(f"Unexpected report image outside approved data/report directories: {source}")


def source_video_specs() -> list[dict[str, Any]]:
    """Resolve exactly the five documented final trimmed EXO/EGO pairs."""
    trim = json.loads(TRIM_MANIFEST.read_text(encoding="utf-8"))
    pairs = {pair["id"]: pair for pair in trim["pairs"]}
    if set(pairs) != {"01", "02", "03", "04", "05"}:
        raise ValueError("Expected exactly the five documented final trimmed pairs")
    specs = []
    for pair_id in sorted(pairs):
        pair = pairs[pair_id]
        for view, field in (("exo", "output_exo"), ("ego", "reference_ego")):
            source = ROOT / pair[field]
            expected_parent = ROOT / "验证视频" / ("裁剪后验证" + view)
            if source.parent != expected_parent or source.suffix.lower() != ".mp4":
                raise ValueError(f"Unexpected documented trimmed source: {source}")
            specs.append({"group": "trimmed_pairs", "pair_id": pair_id, "view": view,
                          "source": source,
                          "path": f"{SOURCE_VIDEO_PREFIX}/trimmed_pairs/pair{pair_id}/{view}.MP4",
                          "source_provenance": "already synchronized and user trimmed; final trimmed 4K paired source",
                          "visible_frame_count": pair["visible_frame_count"],
                          "visible_last_frame_pts_seconds": pair["last_frame_pts"],
                          "provenance_manifest": "evidence/trim_manifest.json"})
    if len(specs) != 10 or len({row["path"] for row in specs}) != 10:
        raise ValueError("Expected exact ten-file final-trimmed 4K whitelist")
    return specs


def stage_source_videos(package: Package) -> dict[str, Any]:
    """Include existing 4K sources byte-for-byte, with explicit processed provenance."""
    specs = source_video_specs()
    source_bytes = sum(spec["source"].stat().st_size for spec in specs)
    if source_bytes >= MAX_ARCHIVE_BYTES:
        raise ValueError("The ten required trimmed 4K sources alone exceed the 2 GiB cap; no source is silently omitted")
    rows = []
    for spec in specs:
        source, target = spec["source"], spec["path"]
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-show_entries",
            "stream=index,codec_type,codec_name,width,height,r_frame_rate,avg_frame_rate,nb_frames,duration,pix_fmt,sample_rate,channels:format=duration,size",
            "-of", "json", str(source)
        ]))
        video = next(stream for stream in probe["streams"] if stream["codec_type"] == "video")
        if (video.get("width"), video.get("height")) != (3840, 2160) or video.get("r_frame_rate") != "25/1":
            raise ValueError(f"Expected documented 3840x2160, 25fps 4K source: {source}")
        package.path_map[str(source)] = target
        package.path_map[project_path(source)] = target
        copied = package.copy_stream(source, target)
        rows.append({"group": spec["group"], "pair_id": spec["pair_id"], "view": spec["view"],
                     "path": target, "source_project_relative_path": project_path(source),
                     "source_provenance": spec["source_provenance"],
                     "provenance_manifest": spec["provenance_manifest"],
                     "visible_frame_count": spec["visible_frame_count"],
                     "visible_last_frame_pts_seconds": spec["visible_last_frame_pts_seconds"],
                     "unprocessed_camera_raw": False, "transcoded_for_packaging": False,
                     "width": video["width"], "height": video["height"],
                     "r_frame_rate": video["r_frame_rate"], "avg_frame_rate": video.get("avg_frame_rate"),
                     "video_codec": video["codec_name"], "pixel_format": video.get("pix_fmt"),
                     "video_stream_duration_seconds": video.get("duration"),
                     "container_duration_seconds": probe.get("format", {}).get("duration"),
                     "declared_frame_count": video.get("nb_frames"),
                     "audio_codecs": ",".join(stream["codec_name"] for stream in probe["streams"] if stream["codec_type"] == "audio"),
                     "sha256": copied["packaged_sha256"], **copied})
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "scope": "Five final trimmed EXO/EGO 4K pairs, with pair04 included once as the actual experiment source",
        "source_video_count": len(rows), "trimmed_pair_count": 5, "source_video_bytes": source_bytes,
        "provenance": "Already synchronized and user-trimmed source files, copied without modification.",
        "frame_count_policy": "MP4 declared frame counts can include hidden preroll; visible counts and last displayed PTS are taken from the existing trim manifest and decode checks, not inferred from nb_frames.",
        "decode_evidence": "evidence/source_videos_4k_decode_checks.json",
        "integrity": "All files copied without transcoding; SHA256 before copy, after copy and on the packaged copy must agree.",
        "files": rows
    }
    package.put(f"{SOURCE_VIDEO_PREFIX}/manifest.json", encode_json(manifest))
    csv_fields = ["group", "pair_id", "view", "path", "source_project_relative_path", "source_provenance",
                  "provenance_manifest", "visible_frame_count", "visible_last_frame_pts_seconds",
                  "unprocessed_camera_raw", "transcoded_for_packaging", "width", "height",
                  "r_frame_rate", "avg_frame_rate", "video_codec", "pixel_format", "video_stream_duration_seconds",
                  "container_duration_seconds", "declared_frame_count", "audio_codecs", "bytes", "sha256",
                  "source_sha256_before", "source_sha256_after", "packaged_sha256", "source_unchanged_after_copy"]
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=csv_fields, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    package.put(f"{SOURCE_VIDEO_PREFIX}/manifest.csv", buffer.getvalue().encode("utf-8"))
    decode_checks = ROOT / "records/user_ego_trim_20261019/decode_checks.json"
    package.copy_json(decode_checks, "evidence/source_videos_4k_decode_checks.json")
    readme_text = f"""# 随包裁剪后4K素材

本目录为最终裁剪后的5组EXO/EGO双视角素材，共10条3840×2160、25fps视频，合计{source_bytes:,}字节。按现有源文件逐字节复制，不转码；MP4在ZIP中使用ZIP_STORED。

`trimmed_pairs/pair01`至`pair05`各含EXO与EGO。配对04就是本实验使用的裁剪源，只收录一次。所有视频均已经同步并按用户区间裁剪，保留各自音轨。960×540工作副本和动态检测成品另见`data/video_pair04/`与`videos/`。

`manifest.json`与`manifest.csv`记录每条视频的原项目相对路径、包内路径、分辨率、帧率、编码、时长、大小及SHA256。MP4可能带不可见的预滚帧，容器或流的`nb_frames`不等于可见画面数；可见帧数按既有裁剪验收分别为365、350、280、311、265帧，详见`evidence/trim_manifest.json`与`evidence/source_videos_4k_decode_checks.json`。容器时长与可见画面时长也可能有少量差异。

源文件复制前后及包内副本SHA256必须一致。整体ZIP另有CRC和逐文件SHA256校验。这里提供的是裁剪后源素材，未重新编码或补帧。
"""
    package.put(f"{SOURCE_VIDEO_PREFIX}/README.md", readme_text.encode("utf-8"))
    return {"source_video_count": len(rows), "source_video_bytes": source_bytes,
            "trimmed_pair_count": 5, "paths": [row["path"] for row in rows],
            "all_source_hashes_unchanged_before_after": all(row["source_sha256_before"] == row["source_sha256_after"] == row["packaged_sha256"] for row in rows),
            "unprocessed_camera_raw_claimed": False, "transcoding_performed": False,
            "manifest": f"{SOURCE_VIDEO_PREFIX}/manifest.json",
            "decode_evidence": "evidence/source_videos_4k_decode_checks.json"}



def stage_paper_figures(package: Package) -> dict[str, Any]:
    """Package final scientific figures and source/QA data, not debug overlays."""
    if not PAPER_FIGURE_SOURCE.is_dir():
        raise FileNotFoundError("Final paper figures are not ready")
    package.path_map[str(PAPER_FIGURE_SOURCE)] = "report/paper_figures"
    package.path_map[project_path(PAPER_FIGURE_SOURCE)] = "report/paper_figures"
    figure_stems = ("figure1_feature_plate", "figure2_stability", "figure3_retention_case")
    for stem in figure_stems:
        for extension in (".png", ".pdf", ".svg"):
            expected = PAPER_FIGURE_SOURCE / (stem + extension)
            if not expected.is_file():
                raise FileNotFoundError(f"Missing final paper figure: {expected}")
    files = []
    for source in sorted(PAPER_FIGURE_SOURCE.rglob("*")):
        if not source.is_file() or source.name.startswith(".") or "__pycache__" in source.parts:
            continue
        if source.suffix.lower() not in (".png", ".pdf", ".svg", ".json", ".csv", ".tsv", ".md", ".txt", ".log", ".py"):
            raise ValueError(f"Unexpected paper-figure file type: {source}")
        if source.suffix.lower() == ".py" and source.parent.name != "qa_tools":
            raise ValueError(f"Only portable figure QA helpers may be nested Python sources: {source}")
        # Collision-marked PDFs are diagnostics, never substitutes for the final figure.
        if source.suffix.lower() == ".pdf" and source.stem not in figure_stems:
            continue
        relative = "report/paper_figures/" + source.relative_to(PAPER_FIGURE_SOURCE).as_posix()
        package.path_map[str(source)] = relative
        package.path_map[project_path(source)] = relative
        files.append((source, relative))
    for source, relative in files:
        if source.suffix.lower() == ".json":
            package.copy_json(source, relative)
        else:
            package.copy(source, relative)
    for name in ("revision_notes.md", "report_revision_review.json", "report_revision_review.md",
                 "report_portable_reproduction.json"):
        source = SOURCE_REPORT / name
        if not source.is_file():
            continue
        if source.suffix == ".json":
            package.copy_json(source, "evidence/" + name)
        else:
            package.copy(source, "evidence/" + name)
    for name in ("paper_revision_review.json", "paper_revision_review.md",
                 "paper_portable_reproduction.json", "manuscript_consistency.json"):
        source = SOURCE_REPORT / name
        if not source.is_file():
            continue
        relative = "evidence/history/" + name
        if source.suffix == ".json":
            historical = package.rewrite_paths(json.loads(source.read_text(encoding="utf-8")))
            if not isinstance(historical, dict):
                historical = {"original_record": historical}
            historical["packaging_history_scope"] = {
                "version": "Historical six-page paper-form revision",
                "current_report_evidence": False,
                "note": "This record preserves the six-page draft's own scripts, figures and report validation; it does not validate the restored process-focused experiment report.",
                "current_validation": "evidence/report_portable_reproduction.json and evidence/delivery_validation.json"
            }
            package.put(relative, encode_json(historical), source,
                        "Historical six-page draft evidence; original declared hashes retained.")
        else:
            package.put(relative, ("# Historical six-page draft record\n\nNot evidence for the current experiment report.\n\n" + source.read_text(encoding="utf-8")).encode("utf-8"), source)
        package.retire("evidence/" + name)
    return {"figure_count": len(figure_stems), "export_formats": ["PNG", "PDF", "SVG"],
            "packaged_figure_and_source_data_files": len(files),
            "redrawn_from_existing_experiment_records": True,
            "new_experiments_claimed": False}


def stage_dynamic(package: Package) -> dict[str, Any]:
    """Include exactly two working inputs and one composed result, never 4K."""
    if not DYNAMIC_SOURCE.is_dir():
        raise FileNotFoundError("Final dynamic outputs are not ready")
    package.path_map[str(DYNAMIC_SOURCE)] = "evidence/dynamic"
    package.path_map[project_path(DYNAMIC_SOURCE)] = "evidence/dynamic"
    probes = {}
    for filename, relative in VIDEO_PATHS.items():
        source = DYNAMIC_SOURCE / filename
        if not source.is_file():
            raise FileNotFoundError(f"Required dynamic work video is missing: {source}")
        package.path_map[str(source)] = relative
        package.path_map[project_path(source)] = relative
        package.path_map[filename] = relative
        probe = json.loads(subprocess.check_output([
            "ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
            "stream=width,height,r_frame_rate,nb_frames,duration", "-of", "json", str(source)
        ]))["streams"][0]
        if filename in ("exo.mp4", "ego.mp4") and (probe["width"], probe["height"]) != (960, 540):
            raise ValueError(f"Only 960x540 dynamic work inputs are allowed: {source}")
        if probe["width"] > 1920 or probe["height"] > 1080:
            raise ValueError(f"Oversized dynamic result is not a permitted work video: {source}")
        probes[relative] = probe
        package.copy(source, relative)
    # Register all small-evidence file paths before rewriting JSON references.
    evidence_files = []
    for source in sorted(DYNAMIC_SOURCE.rglob("*")):
        if not source.is_file() or source.name == ".DS_Store":
            continue
        if source.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi"):
            if source.parent != DYNAMIC_SOURCE or source.name not in VIDEO_PATHS:
                raise ValueError(f"Unexpected fourth video in dynamic outputs: {source}")
            continue
        if source.suffix.lower() not in (".json", ".csv", ".log", ".txt", ".md", ".jpg", ".jpeg", ".png"):
            raise ValueError(f"Unexpected dynamic evidence type: {source}")
        relative = "evidence/dynamic/" + source.relative_to(DYNAMIC_SOURCE).as_posix()
        package.path_map[str(source)] = relative
        package.path_map[project_path(source)] = relative
        package.path_map[source.relative_to(DYNAMIC_SOURCE).as_posix()] = relative
        if source.parent == DYNAMIC_SOURCE:
            package.path_map[source.name] = relative
        evidence_files.append((source, relative))
    for source, relative in evidence_files:
        if source.name == "manifest.json" and source.parent == DYNAMIC_SOURCE:
            package.copy(source, "evidence/dynamic/source_manifest.json")
            portable = package.rewrite_paths(json.loads(source.read_text(encoding="utf-8")))
            portable["packaging_note"] = {
                "run_scope": "This is the original source-project dynamic run, not a claim of a new run inside this package.",
                "original_manifest": "evidence/dynamic/source_manifest.json",
                "portable_copy": "Only file-location strings are mapped to packaged work videos and evidence; original numeric records, timings and source hashes are preserved."
            }
            package.put(relative, encode_json(portable), source,
                        "Portable path copy with original-run note; untouched source manifest is included separately.")
        elif source.suffix == ".json":
            package.copy_json(source, relative)
        else:
            package.copy(source, relative)
    return {"included_video_count": len(probes), "work_video_probes": probes,
            "evidence_file_count": len(evidence_files), "is_feature_tracking": False}


def rewrite_markdown_images(package: Package, text: str) -> str:
    pattern = re.compile(r"(!\[[^\]]*\]\()<?([^>\n\)]+)>?(\))")
    def replace(match: re.Match[str]) -> str:
        target = match[2]
        if target.startswith(("http://", "https://", "data:")):
            return match[0]
        source = Path(target)
        if not source.is_absolute():
            source = SOURCE_REPORT / source
        mapped = image_target(package, source)
        relative = os.path.relpath(TARGET / mapped, TARGET / "report")
        return match[1] + "<" + Path(relative).as_posix() + ">" + match[3]
    return pattern.sub(replace, text)


def finalize(package: Package, report_validation: Path, contact_sheet: Path, delivery_validation: Path) -> dict[str, Any]:
    delivery = json.loads(delivery_validation.read_text(encoding="utf-8"))
    if delivery.get("status") != "passed" or not delivery.get("tested_payload_sha256"):
        raise ValueError("Current delivery validation with tested payload hashes is required")
    for name in ("course_feature_detection.py", "audit_feature_results.py", "assemble_feature_report.py", "build_week08_pdf.py", "build_feature_video.py", "audit_feature_video.py", "plot_week08_paper_figures.py"):
        package.copy(ROOT / "scripts" / name, "scripts/" + name)
    dynamic_inventory = stage_dynamic(package)
    source_video_inventory = stage_source_videos(package)
    allowed_videos = set(VIDEO_PATHS.values()) | set(source_video_inventory["paths"])
    if len(allowed_videos) != 13:
        raise ValueError("Expected exactly ten trimmed 4K sources and three named work videos")
    paper_figure_inventory = stage_paper_figures(package)
    dynamic_audit_source = SOURCE_REPORT / "independent_dynamic_audit.json"
    dynamic_audit = package.rewrite_paths(json.loads(dynamic_audit_source.read_text(encoding="utf-8")))
    dynamic_audit["packaging_note"] = {
        "audit_scope": "Audit of the source-project dynamic artifacts; source manifest/audit hashes are historical original-input values.",
        "original_manifest": "evidence/dynamic/source_manifest.json",
        "portable_copy": "Location strings are mapped for this package, not presented as a new audit run. Use the included audit_feature_video.py to check packaged paths anew.",
        "detector_spot_checks": "Same-function sample redetection, not an independent Harris/SIFT implementation."
    }
    package.put("evidence/independent_dynamic_audit.json", encode_json(dynamic_audit), dynamic_audit_source,
                "Portable locations and explicit source-audit note; original declared input hashes are preserved.")
    result_source = SOURCE_REPORT / "results"
    if not (result_source / "metrics.json").is_file() or not (result_source / "audit.json").is_file():
        raise FileNotFoundError("Final experiment results are not ready")
    metrics = json.loads((result_source / "metrics.json").read_text(encoding="utf-8"))
    audit = json.loads((result_source / "audit.json").read_text(encoding="utf-8"))
    if len(metrics.get("records", [])) != 48 or len(metrics.get("images", [])) != 4 or audit.get("status") != "passed":
        raise ValueError("Expected complete four-image/48-case results with passed audit")
    package.path_map[str(result_source)] = "results"
    package.path_map[project_path(result_source)] = "results"
    for source in sorted(result_source.rglob("*")):
        if not source.is_file() or source.name == ".DS_Store":
            continue
        relative = "results/" + source.relative_to(result_source).as_posix()
        if source.suffix == ".json":
            package.copy_json(source, relative)
        elif source.name == "README.md":
            original = source.read_text(encoding="utf-8")
            portable = re.sub(r"Reproduce from the project root.*", "Reproduce from the submission root with an empty destination:\n\n```sh\npython scripts/course_feature_detection.py --frames-root data/frames/scene04 --output results_reproduced\n```\n", original, flags=re.S)
            package.put(relative, portable.encode("utf-8"), source, "Reproduction command converted to submission-relative paths.")
        else:
            package.copy(source, relative)
    diagnostics_candidates = [SOURCE_REPORT / "diagnostics.json", result_source / "diagnostics.json"]
    diagnostics = next((path for path in diagnostics_candidates if path.is_file()), None)
    if diagnostics is None:
        raise FileNotFoundError("Expected final diagnostics.json from report assembly")
    package.copy_json(diagnostics, "results/diagnostics.json")
    package.copy_json(diagnostics, "evidence/diagnostics.json")
    package.copy_json(result_source / "report_summary.json", "evidence/report_summary.json")
    independent_source = SOURCE_REPORT / "independent_audit.json"
    independent = package.rewrite_paths(json.loads(independent_source.read_text(encoding="utf-8")))
    independent["packaging_note"] = {
        "audit_scope": "Historical independent audit of the original source-project files before portable path rewriting.",
        "hash_meaning": "The declared metrics_sha256 and producer_audit_sha256 remain the original source-file hashes; they are not replaced with hashes of packaged files that were not the original audit inputs.",
        "portable_paths": "Only location strings are rewritten; frames_root is data/frames/scene04. Numeric results and audit conclusions are preserved.",
        "hash_mapping": "manifest.json source_to_submission_mappings records the original and packaged SHA256 values separately.",
        "reaudit": "Run scripts/audit_feature_results.py on results with --frames-root data/frames/scene04 to audit packaged files anew. Current delivery evidence is evidence/delivery_validation.json; previous static reproduction is explicitly historical."
    }
    package.put("evidence/independent_audit.json", encode_json(independent), independent_source,
                "Portable location strings plus explicit historical-audit note; original declared audit-input hashes are preserved.")
    report_json = SOURCE_REPORT / "report_data.json"
    data = json.loads(report_json.read_text(encoding="utf-8"))
    for block in data.get("blocks", []):
        if block.get("type") != "image":
            continue
        source = Path(block["path"])
        if not source.is_absolute():
            source = ROOT / source
        relative = image_target(package, source)
        if not source.is_file() or source.suffix.lower() not in (".jpg", ".jpeg", ".png", ".webp"):
            raise FileNotFoundError(f"Missing/unsupported report image: {source}")
        package.copy(source, relative)
        package.path_map[str(source.resolve())] = relative
        package.path_map[project_path(source)] = relative
    package.put("report/report_data.json", encode_json(package.rewrite_paths(data)), report_json,
                "Report image paths rewritten to submission-root relative paths; facts and numbers unchanged.")
    source_md = SOURCE_REPORT / "实验报告.md"
    package.put("report/实验报告.md", rewrite_markdown_images(package, source_md.read_text(encoding="utf-8")).encode("utf-8"), source_md,
                "Markdown image links rewritten relative to report/.")
    pdf_source = SOURCE_REPORT / "第8周_Harris与SIFT实验报告.pdf"
    package.copy(pdf_source, "report/第8周_Harris与SIFT实验报告.pdf")
    for source, relative in ((report_json, "report/report_data.json"), (source_md, "report/实验报告.md"),
                              (pdf_source, "report/第8周_Harris与SIFT实验报告.pdf")):
        package.path_map[str(source.resolve())] = relative
        package.path_map[project_path(source)] = relative
    package.copy_json(report_validation, "evidence/report_validation.json")
    package.copy(contact_sheet, "evidence/contact_sheet.jpg")
    reproduction_check = SOURCE_REPORT / "reproduction_check.json"
    static_unchanged_payload = {}
    if reproduction_check.is_file():
        original_static = json.loads(reproduction_check.read_text(encoding="utf-8"))
        historical_static = package.rewrite_paths(original_static)
        historical_static["packaging_note"] = {
            "version_scope": "Previous static-only release before dynamic video and report updates.",
            "hash_scope": "Its tested script/report hashes are historical and are not evidence for the revised assembler or current delivery.",
            "unchanged_core": "Four JPEG inputs, static metrics.json and 24 coordinate files remain byte-identical; they are checked separately during current packaging.",
            "current_evidence": "evidence/delivery_validation.json"
        }
        package.put("evidence/history/static_reproduction_check.json", encode_json(historical_static), reproduction_check,
                    "Previous-version reproduction preserved as history; no old script hash is claimed for this revised delivery.")
        package.retire("evidence/reproduction_check.json")
        static_unchanged_payload = {
            path: value for path, value in original_static.get("tested_payload_sha256", {}).items()
            if path.startswith("data/frames/") or path == "results/metrics.json" or path.startswith("results/coordinates/")
        }
        if len(static_unchanged_payload) != 29:
            raise ValueError("Expected four inputs + static metrics + 24 coordinates in historical invariants")
    package.copy_json(delivery_validation, "evidence/delivery_validation.json")
    # Git is local metadata only. No identity, commit, remote or network action.
    subprocess.run(["git", "init", "-q", str(TARGET)], check=True)
    payload_paths = sorted(path for path in package.state["managed"] if path not in ("manifest.json", "SHA256SUMS"))
    for relative in payload_paths:
        path = TARGET / relative
        if sha256(path) != package.state["managed"][relative]:
            raise ValueError(f"Managed file changed during packaging: {relative}")
        if path.suffix.lower() in (".mp4", ".mov", ".mkv", ".avi") and relative not in allowed_videos:
            raise ValueError(f"Video outside the exact 13-file whitelist: {relative}")
        if path.suffix.lower() in (".safetensors", ".ckpt", ".pt"):
            raise ValueError(f"Model data must not enter this submission: {relative}")
    actual_videos = {relative for relative in payload_paths if Path(relative).suffix.lower() in (".mp4", ".mov", ".mkv", ".avi")}
    if actual_videos != allowed_videos:
        raise ValueError("Packaged video inventory differs from the exact 13-file whitelist")
    manifest = {
        "schema_version": 1, "created_utc": datetime.now(timezone.utc).isoformat(),
        "repository_kind": "local git only; no commit, identity or remote created by the packaging tool",
        "source_project_relative_path": "submission/week08_harris_sift",
        "package_root_name": TARGET.name,
        "included_video_count": len(allowed_videos), "included_work_video_count": 3,
        "included_source_video_count": source_video_inventory["source_video_count"],
        "included_source_video_bytes": source_video_inventory["source_video_bytes"],
        "included_model_count": 0, "archive_size_cap_bytes": MAX_ARCHIVE_BYTES,
        "video_zip_compression": "ZIP_STORED; no transcoding",
        "input_jpeg_count": 4, "pair_id": "04", "field_scene_id": "2",
        "scope": "Week08 process-focused Harris/SIFT experiment report with abstract and keywords, full controlled comparison, supplementary scientific figures and one paired work-video demonstration; framewise detection is not tracking. No cloud, training, self-captured geometry or EgoX inference.",
        "dynamic_inventory": dynamic_inventory,
        "source_video_inventory": source_video_inventory,
        "paper_figure_inventory": paper_figure_inventory,
        "previous_static_core_unchanged_file_count": len(static_unchanged_payload),
        "hash_policy": "Files lists payload except manifest/SHA256SUMS; SHA256SUMS includes manifest, excludes itself and .git/.package_state.json.",
        "source_to_submission_mappings": package.state["sources"],
        "rewritten_path_mappings": package.path_map,
        "files": [{"path": relative, "sha256": sha256(TARGET / relative), "bytes": (TARGET / relative).stat().st_size} for relative in payload_paths],
    }
    # Store source paths as project-relative, including the rewrite-map keys.
    manifest["rewritten_path_mappings"] = {
        key[len(str(ROOT)) + 1:] if key.startswith(str(ROOT) + "/") else key: value
        for key, value in package.path_map.items()
    }
    package.put("manifest.json", encode_json(manifest))
    hash_paths = sorted(payload_paths + ["manifest.json"])
    sums = "".join(sha256(TARGET / path) + "  " + path + "\n" for path in hash_paths)
    package.put("SHA256SUMS", sums.encode("utf-8"))
    package.save_state()
    archive_paths = sorted(hash_paths + ["SHA256SUMS"])
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    partial = ARCHIVE.with_name(ARCHIVE.stem + ".partial.zip")
    with zipfile.ZipFile(partial, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as archive:
        for relative in archive_paths:
            compression = zipfile.ZIP_STORED if Path(relative).suffix.lower() == ".mp4" else zipfile.ZIP_DEFLATED
            archive.write(TARGET / relative, arcname=TARGET.name + "/" + relative, compress_type=compression)
    if partial.stat().st_size >= MAX_ARCHIVE_BYTES:
        partial.unlink()
        raise ValueError("ZIP exceeds the 2 GiB package cap; no required video is silently omitted")
    with zipfile.ZipFile(partial) as archive:
        failure = archive.testzip()
        if failure:
            raise ValueError(f"ZIP CRC check failed: {failure}")
        expected = {TARGET.name + "/" + path for path in archive_paths}
        if set(archive.namelist()) != expected:
            raise ValueError("ZIP file inventory differs from managed payload")
        for line in sums.splitlines():
            expected_hash, relative = line.split("  ", 1)
            with archive.open(TARGET.name + "/" + relative) as handle:
                actual_hash = stream_digest(handle)
            if actual_hash != expected_hash:
                raise ValueError(f"ZIP SHA256 mismatch: {relative}")
        for relative, expected_hash in static_unchanged_payload.items():
            with archive.open(TARGET.name + "/" + relative) as handle:
                actual_hash = stream_digest(handle)
            if actual_hash != expected_hash:
                raise ValueError(f"Previously validated static input/numeric data changed: {relative}")
        tested_payload = delivery["tested_payload_sha256"]
        for relative, expected_hash in tested_payload.items():
            with archive.open(TARGET.name + "/" + relative) as handle:
                actual_hash = stream_digest(handle)
            if actual_hash != expected_hash:
                raise ValueError(f"Final ZIP differs from current delivery-validated payload: {relative}")
        for relative in allowed_videos:
            if archive.getinfo(TARGET.name + "/" + relative).compress_type != zipfile.ZIP_STORED:
                raise ValueError(f"Video was unexpectedly ZIP-compressed: {relative}")
    for spec in source_video_specs():
        if sha256(spec["source"]) != package.state["sources"][spec["path"]]["source_sha256_before"]:
            raise ValueError(f"Source video changed between staging and final ZIP verification: {spec['source']}")
    os.replace(partial, ARCHIVE)
    archive_sha256 = sha256(ARCHIVE)
    ARCHIVE.with_name(ARCHIVE.name + ".sha256").write_text(archive_sha256 + "  " + ARCHIVE.name + "\n", encoding="utf-8")
    return {"status": "verified", "repository": project_path(TARGET), "zip": project_path(ARCHIVE),
            "zip_sha256": archive_sha256, "zip_bytes": ARCHIVE.stat().st_size,
            "zip_files": len(archive_paths), "crc_and_sha256_checks": "passed",
            "current_delivery_tested_payload_files": len(tested_payload),
            "current_delivery_tested_payload_check": "passed",
            "previous_static_input_numeric_files_unchanged": len(static_unchanged_payload),
            "included_work_video_count": 3,
            "included_source_video_count": source_video_inventory["source_video_count"],
            "included_source_video_bytes": source_video_inventory["source_video_bytes"],
            "included_video_count": len(allowed_videos),
            "source_hashes_unchanged_before_after_zip": True,
            "all_mp4_zip_stored": True,
            "git_metadata_in_zip": False}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    choice = parser.add_mutually_exclusive_group(required=True)
    choice.add_argument("--prepare", action="store_true")
    choice.add_argument("--finalize", action="store_true")
    parser.add_argument("--report-validation", type=Path, default=SOURCE_REPORT / "report_validation.json")
    parser.add_argument("--contact-sheet", type=Path, default=SOURCE_REPORT / "contact_sheet.jpg")
    parser.add_argument("--delivery-validation", type=Path, default=SOURCE_REPORT / "delivery_validation.json")
    args = parser.parse_args()
    package = Package()
    prepare(package)
    if args.finalize:
        result = finalize(package, args.report_validation.resolve(), args.contact_sheet.resolve(), args.delivery_validation.resolve())
        (ARCHIVE.parent / "week08_package_validation.json").write_bytes(encode_json(result))
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(json.dumps({"status": "prepared_only", "repository": project_path(TARGET),
                          "managed_files": len(package.state["managed"]), "zip_written": False,
                          "git_metadata_exists": (TARGET / ".git").exists(),
                          "git_initialized_by_this_prepare": False}, ensure_ascii=False))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Assemble the Week 8 report from measured Harris/SIFT outputs, never invented scores."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import re
from statistics import mean

from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
LABELS = {"identity": "原图恒等对照", "rotation15": "旋转15°", "rotation45": "旋转45°", "scale075": "缩小至0.75倍", "scale125": "放大至1.25倍", "gaussian": "高斯模糊"}


def resolve(path: Path) -> Path:
    return path if path.is_absolute() else ROOT / path


def relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(ROOT).as_posix()
    except ValueError:
        return str(path.resolve())


def font(size: int):
    for path in ["/System/Library/Fonts/Monaco.ttf", "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf", "/usr/share/fonts/truetype/liberation2/LiberationMono-Regular.ttf"]:
        if Path(path).is_file():
            return ImageFont.truetype(path, size)
    return ImageFont.load_default(size=size)


def make_assets(results: Path, assets: Path) -> None:
    assets.mkdir(parents=True, exist_ok=True)
    for view in ["exo", "ego"]:
        with Image.open(results / f"overview_{view}.png") as source:
            # Rearrange the first original/heat/corner/SIFT row into a legible 2x2.
            board = Image.new("RGB", (960, 628), "#18212b")
            for column in range(4):
                board.paste(source.crop((column * 480, 0, (column + 1) * 480, 314)), ((column % 2) * 480, (column // 2) * 314))
            board.save(assets / f"feature_{view}_f50.png")
    originals = Image.new("RGB", (960, 314), "#18212b")
    for column, view in enumerate(["exo", "ego"]):
        with Image.open(results / f"overview_{view}.png") as source:
            originals.paste(source.crop((0, 0, 480, 314)), (column * 480, 0))
    originals.save(assets / "paired_originals.png")
    conditions = ["rotation45", "scale075", "gaussian"]
    gallery = Image.new("RGB", (960, 436), "#18212b")
    labels = {"rotation45": "rotation 45 deg", "scale075": "scale 0.75", "gaussian": "blur 9x9, sigma=2"}
    d = ImageDraw.Draw(gallery)
    for row, view in enumerate(["exo", "ego"]):
        for column, condition in enumerate(conditions):
            x, y = 320 * column, 218 * row
            d.text((x + 6, y + 8), f"{view.upper()} | {labels[condition]}", font=font(16), fill="white")
            with Image.open(results / "transformed_images" / f"{view}_f0050" / f"{condition}.jpg") as image:
                gallery.paste(image.resize((320, 180), Image.Resampling.LANCZOS), (x, y + 38))
    gallery.save(assets / "controlled_samples.png")
    # Show the diagnosed F200 case, retaining green/red positions at full 960x540.
    panel = Image.new("RGB", (960, 628), "#18212b")
    draw = ImageDraw.Draw(panel)
    case = json.loads((results / "coordinates/ego_f0200/rotation45.json").read_text())
    with Image.open(results / "transformed_images/ego_f0200/rotation45.jpg") as source:
        for column, method in enumerate(["harris", "sift"]):
            image = source.copy().convert("RGB")
            ed = ImageDraw.Draw(image)
            data = case["methods"][method]
            common = data["common_target_retained_indices"]
            matches = {int(m[1]) for m in data["matching"]}
            for i, full_index in enumerate(common):
                x, y = data["target_retained_positions"][full_index][:2]
                color = "#31e443" if i in matches else "#ff5142"
                ed.ellipse((x - 4, y - 4, x + 4, y + 4), outline=color, width=2)
            image = image.resize((480, 270), Image.Resampling.LANCZOS)
            panel.paste(image, (column * 480, 44))
            draw.text((column * 480 + 8, 12), f"{method.upper()} | common target centers", fill="white", font=font(17))
    # Second row: the observed blur case, with all retained target positions.
    case = json.loads((results / "coordinates/ego_f0200/gaussian.json").read_text())
    with Image.open(results / "transformed_images/ego_f0200/gaussian.jpg") as source:
        for column, method in enumerate(["harris", "sift"]):
            image = source.copy().convert("RGB")
            ed = ImageDraw.Draw(image)
            for x, y, *_ in case["methods"][method]["target_retained_positions"]:
                ed.ellipse((x - 4, y - 4, x + 4, y + 4), outline="#ffd83b", width=2)
            panel.paste(image.resize((480, 270), Image.Resampling.LANCZOS), (column * 480, 358))
            draw.text((column * 480 + 8, 326), f"{method.upper()} | blurred target centers", fill="white", font=font(17))
    panel.save(assets / "failure_case.png")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results", type=Path, default=Path("output/week08_harris_sift/results"))
    parser.add_argument("--output", type=Path, default=Path("output/week08_harris_sift/report_data.json"))
    parser.add_argument("--assets", type=Path, help="Default: assets/ next to report JSON")
    parser.add_argument("--capture-evidence", type=Path, default=Path("output/week08/report_assets/capture_setup.jpg"))
    parser.add_argument("--paper-figures", type=Path, help="Optional supplementary scientific figure directory")
    parser.add_argument("--dynamic", type=Path, help="Optional directory containing the dynamic video manifest, counts, audit and previews")
    args = parser.parse_args()
    results, output, capture = resolve(args.results), resolve(args.output), resolve(args.capture_evidence)
    assets = resolve(args.assets) if args.assets else output.parent / "assets"
    metrics = json.loads((results / "metrics.json").read_text())
    audit = json.loads((results / "audit.json").read_text())
    assert audit["status"] == "passed" and len(metrics["records"]) == 48
    assert metrics["opencv_version"] == "4.11.0"
    records = metrics["records"]
    default_dynamic = ROOT / "output/week08_harris_sift/dynamic"
    dynamic = resolve(args.dynamic) if args.dynamic else (default_dynamic if (default_dynamic / "audit.json").is_file() else None)
    dynamic_manifest, dynamic_rows = None, []
    if dynamic is not None:
        dynamic_manifest = json.loads((dynamic / "manifest.json").read_text())
        dynamic_audit = json.loads((dynamic / "audit.json").read_text())
        assert dynamic_audit["status"] == "passed"
        with (dynamic / "counts.csv").open(newline="", encoding="utf-8") as handle:
            dynamic_rows = list(csv.DictReader(handle))
        assert len(dynamic_rows) == 4 * dynamic_manifest["frame_count"]
    make_assets(results, assets)
    get = lambda image, cond, method: next(r for r in records if r["image_id"] == image and r["condition"] == cond and r["method"] == method)
    pct = lambda n: f"{100*n:.1f}%"
    diagnostics = []
    for r in records:
        ns, nt, m = r["common_source_count"], r["common_target_count"], r["one_to_one_match_count"]
        diagnostics.append({"image_id": r["image_id"], "condition": r["condition"], "method": r["method"], "Ns": ns, "Nt": nt, "M": m, "R_min": r["repeatability"], "source_coverage": m/ns if ns else None, "target_coverage": m/nt if nt else None})
    output.parent.mkdir(parents=True, exist_ok=True)
    (output.parent / "diagnostics.json").write_text(json.dumps({"purpose": "Explain point-count imbalance; descriptive diagnostics, not additional detector training or semantic accuracy", "records": diagnostics}, ensure_ascii=False, indent=2) + "\n")

    blocks = []
    section_count, figure_count = 0, 0
    def text(kind, value):
        nonlocal section_count
        if kind == "h1":
            section_count += 1
            value = re.sub(r"^\d+\s+", f"{section_count} ", value)
        blocks.append({"type": kind, "text": value})
    def table(headers, rows, widths=None):
        blocks.append({"type": "table", "headers": headers, "rows": rows, **({"widths": widths} if widths else {})})
    def picture(path, caption, height=260):
        nonlocal figure_count
        figure_count += 1
        caption = re.sub(r"^图\d+", f"图{figure_count}", caption)
        blocks.append({"type": "image", "path": relative(path), "caption": caption, "width": 490, "max_height": height})
    def page(): blocks.append({"type": "pagebreak"})

    paper_figures = resolve(args.paper_figures) if args.paper_figures else output.parent / "paper_figures"
    text("h2", "摘要")
    text("p", "本实验围绕自采足球传接球视频，完成R8第三人称与Nano第一人称画面的采集、同步、裁剪和特征检测。选取两个时刻、两个视角的4张关键帧，运行Harris与SIFT，并通过旋转、缩放和模糊开展48组受控对照；另提供12.44秒双视角动态检测视频。报告展示原图、响应图、特征分布、变换样例、完整结果表及失败案例。结果表明，画面内容与扰动会影响特征数量和位置保留；评价时应结合有效对应数与源点保留，不能仅凭点数或重复率判断。")
    text("p", "关键词：足球自采视频；双视角；Harris；SIFT；特征稳定性")

    # 1. One narrative: football data -> usable features -> stable evidence -> use.
    text("h1", "1 实验目的与整体流程")
    text("p", "本项目希望利用固定Canon R8与球员佩戴的DJI Nano，研究足球传接球的双视角视频，并为后续第一人称视角转换准备素材。第8周先回答一个基础问题：两种画面中能检测到哪些局部特征？当画面发生旋转、尺度变化或模糊时，这些位置还能被重复找到吗？")
    picture(assets / "paired_originals.png", "图1 同一时刻F50：R8场边视角（左）与Nano实拍视角（右）。两者时间同步，空间投影不同。", 165)
    text("p", "先对自采配对04的完整同步短片逐帧检测，观察动作过程中的特征分布；再对两个时刻、两个视角共4张关键帧施加已知变化，分析位置稳定性与点集保留情况。" if dynamic else "以自采配对04的两个时刻、两个视角共4张工作图为输入，实际运行Harris和SIFT；用已知图像变换做可核查的对照。实验重点是位置稳定性与点集保留情况，而非比较谁检测的点更多。")
    text("p", "整体流程：R8/Nano采集 → 拍掌同步与对应裁剪 → 选取关键帧 → 生成受控变化 → Harris/SIFT检测 → 图表比较与失败分析。后文先展示素材与实际检测结果，再给出评价方法和定量对照。")

    # 2. Acquisition stays concise and evidence-grounded.
    page()
    text("h1", "2 数据采集与同步处理")
    text("p", "2026年10月19日，围绕小范围足球传接球采集R8/Nano配对素材。R8位于场边三脚架上，Nano随佩戴者移动。本次使用配对04（文件目录scene04，对应现场场景2）；它是一个短片中的两个时间点，不是4段独立录像。")
    table(["项目", "已核实的信息", "用途与边界"], [
        ["视频", "两路3840×2160、25fps、横屏", "统一时间轴；编码格式可不同。"],
        ["R8设置照片", "24mm、1/60s、Tv模式", "设置照片证据；未逐条核实ISO/光圈。"],
        ["现场记录", "相机高度1.23m；近碟约6.3m", "位置记录，不能代替完整相机标定。"],
        ["Nano", "头部附近佩戴；实拍参考", "精确FOV、防抖状态和内参尚未确定。"],
    ], [1.15, 2.6, 2.7])
    picture(capture, "图2 现场机位记录。拍摄设置另见随包元数据；设置照片和空间测量不能代替完整标定。", 145)
    text("h2", "先对齐时间，再选取工作图")
    text("p", "已有同步流程依据拍掌音频定位，再核对拍掌画面与中间动作；视觉同步不确定度约±1帧（25fps下约40ms）。之后按用户裁剪的EGO区间提取对应EXO，排除负PTS/预滚帧后，以可见帧序号选图。设备日期相差18分钟不用于精确同步。")
    table(["工作帧（从0计）", "短片时间", "基础同步视频帧", "提交的输入"], [["F50", "2.000s", "149", "EXO＋EGO各1张"], ["F200", "8.000s", "299", "EXO＋EGO各1张"]], [1.5, 1.3, 1.7, 2])
    text("p", "工作图统一缩至960×540，保留原始JPEG字节与SHA256。同步只保证时间对应，不把EXO/EGO直接用单应矩阵配准。提交包附5组裁剪后的4K配对素材（10条）、两路960×540工作视频及一条检测成品。4K素材已同步和裁剪，保留来源映射与SHA256；配对04为本次实验输入。" if dynamic else "工作图统一缩至960×540，保留原始JPEG字节与SHA256。同步只保证时间对应；本实验分别在每个视角内做受控变换，不把EXO/EGO直接用单应矩阵配准。提交包附4张实验输入、映射与同步证据，不附体积较大的4K视频。")

    if dynamic:
        page()
        text("h1", "3 动态实验：双视角检测对照")
        n, fps = dynamic_manifest["frame_count"], dynamic_manifest["fps"]
        duration = dynamic_manifest["display_duration_seconds"]
        text("p", f"使用配对04完整同步短片：每路{n}帧、{fps:g}fps，播放{duration:.2f}秒。每一帧分别运行Harris和SIFT，将R8与Nano按同一时间展示，保留动作前后的完整过程。")
        picture(dynamic / "previews/frame_0155.jpg", "图3 动态视频F155（6.20秒）截帧：左列R8、右列Nano，上排Harris、下排SIFT；每格标明保留位置数。完整MP4随仓库提供。", 310)
        table(["方面", "执行方式"], [["时间", f"两路与成品均为{n}帧，F0–F{n-1}，时间与帧号共同显示。"], ["参数", "沿用静态实验；5px最小间距、最多500个位置，每帧重新检测。"], ["复现材料", "两路工作副本、四宫格成品、脚本、逐帧CSV和校验记录。"]], [1, 5])
        text("p", "固定机位和佩戴视角呈现不同的运动与画面内容。视频便于观察脚、球与草地附近的特征分布怎样变化；点数达到500时受预算限制，数量相近并不证明是同一批位置。")
        text("p", "这是逐帧检测展示，尚未建立帧间对应，不能称为跟踪，也不在真实运动视频上直接计算受控实验的重复率。工作视频经过缩放与H264编码，与静态JPEG的像素不同，点数无需完全相同。下面继续用关键帧和已知变化作定量分析。")

    # 3. Enough method detail to reproduce, without turning the assignment into a benchmark.
    page()
    text("h1", "3 实验原理与参数设置")
    text("h2", "两种检测器，统一的位置筛选规则")
    text("p", "Harris利用局部灰度梯度的二阶矩阵A，计算响应R_H=det(A)-k·trace(A)²。两个方向都变化明显的位置通常有较强正响应。SIFT在多尺度空间中检测DoG极值，并给出尺度和方向；本实验调用detect，只比较关键点中心，不测试描述子或特征匹配性能。[1–3,5]")
    table(["部分", "固定设置"], [
        ["输入", "960×540，BGR转灰度；Harris另转float32并除以255。"],
        ["Harris", "blockSize=2，ksize=3，k=0.04；阈值为有效区最大正响应的1%；3×3局部极大值。"],
        ["SIFT", "nfeatures=500，nOctaveLayers=3，contrastThreshold=0.04，edgeThreshold=10，sigma=1.6。"],
        ["共同筛选", "中心按0.001px去重，按响应降序；最小点距5px，最多保留500个位置。"],
    ], [1.25, 5.4])
    text("p", "SIFT同一中心的多方向副本只算一个位置。500是上限，不强行补足；SIFT内部限额在空间去重前已生效，因此两种方法最终点数可以不同。Harris响应阈值随每张图的最大值重新计算，这也是点数变化的可能因素。")
    text("h2", "受控生成：一次只改变一个因素")
    table(["条件", "操作与真值"], [["恒等对照", "保持原图，H为单位矩阵。"], ["旋转", "绕图像中心分别旋转15°、45°。"], ["缩放", "绕中心分别缩小至0.75倍、放大至1.25倍。"], ["模糊", "9×9高斯核，sigma=2，H为单位矩阵。"]], [1.25, 5.4])
    text("p", "输出仍为960×540。几何变化采用双线性插值，越界填黑；保存精确的源→目标3×3矩阵H。有效区由源图掩膜随H变换，再做33×33腐蚀，排除16px图边及填黑区域。源点和目标点都须通过正、逆映射的共同可见性检查。")
    text("p", "这些是图像平面中的受控样本生成，不是新拍摄的真实视角，也不是EgoX生成。特征检测在内存中的变换图上完成；导出的变换JPEG仅供查看，复现须从4张原始工作图重新生成。")

    # 4. Clear visuals, not a wall of small dots.
    page()
    text("h1", "4 静态实验结果：响应与特征分布")
    picture(assets / "feature_exo_f50.png", "图3 EXO F50：原图、Harris正响应（对数显示）、Harris角点与SIFT关键点。热图每张独立归一化，颜色不表示跨图绝对响应强弱。", 260)
    picture(assets / "feature_ego_f50.png", "图4 EGO F50使用完全相同参数。红点为Harris位置；SIFT圆圈与短线表示检测器提供的尺度和方向，但指标仅评估中心。", 260)
    text("p", "EXO包含树木、建筑与草地纹理；EGO更接近脚和球，但常由草地占据大部分画面。背景上的特征不等于球的特征；关键点可用于后续几何研究，但本次尚未完成球检测或跟踪。")

    # 5. Count versus repeatability, with a simple explicit metric.
    page()
    text("h1", "5 受控变换与评价方法")
    picture(assets / "controlled_samples.png", "图5 F50的三种代表性变化，上排EXO、下排EGO。全部5种变化与两时刻完整图在仓库results中保留。", 230)
    text("h2", "用已知H检查位置，不从匹配结果反推真值")
    text("p", "将源点按H投到目标图，在共同可见区域内寻找距离不超过3个目标图像素的目标点。采用最大基数一对一对应，每个位置最多使用一次，避免多个点重复占用同一目标。记共同可见源点数Ns、目标点数Nt、对应数M，则：")
    text("p", "位置重复率 r=M/min(Ns,Nt)；源点保留比例cs=M/Ns。若分母为0，结果记为null。每张图先算比率，再对4图做等权宏平均，不把全部点混在一起计算。")
    text("p", "此设计借鉴经典局部特征评估中“已知变换、共同视域、比率与绝对数并报”的思路。[4] 本作业采用中心距离，而经典区域评估使用区域重叠；二者指标不可直接比较。3px按目标图坐标固定，缩放前后的实际源图容差并不相同。")
    table(["源图", "Harris阈值像素→NMS中心→保留", "SIFT关键点→去重中心→保留"], [[r["image_id"], f"{get(r['image_id'],'identity','harris')['source_raw_count']} → {get(r['image_id'],'identity','harris')['source_unique_count']} → {get(r['image_id'],'identity','harris')['source_retained_count']}", f"{get(r['image_id'],'identity','sift')['source_raw_count']} → {get(r['image_id'],'identity','sift')['source_unique_count']} → {get(r['image_id'],'identity','sift')['source_retained_count']}"] for r in metrics["images"]], [1.4, 2.9, 2.7])
    text("p", "两列原始计数的定义不同，不能横向比较原始数值的大小。SIFT去掉多方向副本后数量下降，是统计口径的修正，不代表检测丢失。")

    # 6. Numbers come from exported records.
    page()
    text("h1", "6 定量实验结果与比较")
    rows = []
    for condition in LABELS:
        group = lambda method: [r for r in records if r["condition"] == condition and r["method"] == method]
        hr, sr = group("harris"), group("sift")
        rows.append([LABELS[condition], pct(mean(r["repeatability"] for r in hr)), pct(mean(r["repeatability"] for r in sr)), f"{mean(r['one_to_one_match_count'] for r in hr):.1f}", f"{mean(r['one_to_one_match_count'] for r in sr):.1f}"])
    table(["条件", "Harris r", "SIFT r", "Harris平均M", "SIFT平均M"], rows, [1.65, 1.2, 1.2, 1.5, 1.5])
    if (paper_figures / "figure2_stability.png").is_file():
        picture(paper_figures / "figure2_stability.png", "图6 上：位置重复率r；下：源点保留比例cs。空心点表示4张相关工作图，短横线为等权宏平均；阈值为3个目标像素。上表补充平均对应数M，不作独立场次的统计推断。", 300)
    else:
        picture(results / "repeatability_summary.png", "图6 4张选定图片的重复率宏平均，阈值为3个目标像素。平均M可为小数；每张图的实际对应数均为整数。", 249)
    text("p", "恒等对照下，两种方法的8个结果均为100%，验证相同输入能一致检测。固定规则下，Harris在两种旋转和缩放条件的r较高；高斯模糊时，两种方法的r均明显低于恒等对照，SIFT的四图平均r为52.4%，Harris为44.4%。")
    text("p", "这些值描述本次4图，不能据此作方法的通用排名。两种检测器原始点集密度和数量不同，空间筛选、阈值与共同视域也会影响结果；尤其当较小点集显著缩小时，r可能仍很高。下一页用真实个例解释这一点。")

    # 7. One memorable finding, not a suite of invented metrics.
    page()
    text("h1", "7 失败案例与结果分析")
    text("p", "观察EGO F200。旋转45°后，Harris有392个共同可见源点、70个目标点，其中68个对应。该重复率为68/70=97.1%，源点保留比例则为68/392=17.3%。SIFT虽然r较低，但找到162个对应，原有点保留比例为63.8%。")
    table(["EGO F200", "方法", "Ns", "Nt", "M", "r", "cs=M/Ns"], [[LABELS[c], a.upper(), str(get("ego_f0200",c,a)["common_source_count"]), str(get("ego_f0200",c,a)["common_target_count"]), str(get("ego_f0200",c,a)["one_to_one_match_count"]), pct(get("ego_f0200",c,a)["repeatability"]), pct(get("ego_f0200",c,a)["one_to_one_match_count"]/get("ego_f0200",c,a)["common_source_count"])] for c in ["rotation45", "gaussian"] for a in ["harris", "sift"]], [1.3, 1.05, .7, .7, .65, 1, 1])
    picture(assets / "failure_case.png", "图7 上排：旋转45°目标图中的共同可见位置，绿为有对应、红为无对应；下排：模糊后的全部保留位置（黄）。左Harris、右SIFT。", 275)
    text("p", "模糊案例更直接：Harris的496个源点只对应28个，SIFT的333个源点只对应27个。SIFT此时r仍为62.8%，但源点保留比例只有8.1%。因此，实用判断应把r、Ns/Nt和M一起看，不能只报一个好看的百分比。")
    text("p", "这个案例是结果后的诊断，不是重新调参得到的改善。草地纹理、图像插值、响应阈值变化与点数预算都可能影响点集；本实验没有分离这些因素的因果贡献。")

    # 8. Close the loop to the project, reproducibility, and credible limits.
    page()
    text("h1", "8 实验总结、局限与复现")
    text("p", "本作业的特色，是把通用检测器放进自采足球双视角数据的实际问题中：先保留采集与同步证据，再用已知变化对照，最后从一个高分反例识别指标局限。这是实验构思与应用分析上的探索，不声称提出新的Harris/SIFT算法。")
    table(["从画面和实验得到的判断", "对后续工作的实际帮助"], [["EXO固定画面含大量背景特征", "可作为几何处理的候选材料，但应另行核验人、球及动态遮挡。"], ["EGO更接近脚和球，草地仍占大面积", "不能以全图关键点数量代替足球相关内容的可见性检查。"], ["模糊后对应点显著减少", "筛片时兼看清晰度、有效对应数和点集保留情况。"], ["两视角时间同步，空间投影不同", "本期不直接做跨视角像素比较，也不假定单个H解释整个真实场景。"]], [2.4, 3.6])
    text("h2", "完成了什么，哪些结论还不能下")
    text("p", f"已完成48组受控实验与{len(dynamic_rows)}次帧内检测。静态结果由独立实现依据导出坐标复核H、有效域、对应数及宏平均；动态视频核验帧数、时间戳、计数与完整解码。完整参数、数据和证据随仓库保存。" if dynamic else "已完成4图×6条件×2方法共48组实际检测与对应分析。生产脚本检查8个恒等对照、变换往返、可见性、点距、最大一对一对应及CSV宏平均；另用不导入生产脚本的独立实现，依据导出坐标复核H、有效域、对应数和宏平均。独立审计未重跑检测器。完整证据随代码保存。")
    text("p", "受控指标只来自一个短片的4张关键帧，完整视频也未增加独立拍摄场次，不能作统计泛化结论。不评价跟踪、描述子、语义检测或三维重建。16px只约束中心，不保证大尺度SIFT支持区完全可见；计时仅作运行记录。" if dynamic else "本实验只有一个短片的两个时间点，两个视角也不是独立拍摄场次；结果属于描述性小实验，不代表足球数据上的统计泛化。不评价SIFT描述子、尺度/方向精度、语义检测或真实跨视角配准。16px中心边界也不保证大尺度SIFT支持区域完全避开无效像素；单次CPU计时只作运行记录。")
    text("p", "第14周再验证单目深度辅助的点云、几何先验与EgoX第一视角生成。本期未运行自采EgoX，也未训练或微调；Nano保持独立实拍参考。Harris/SIFT结果不作为EgoX性能或三维重建成功的证据。")
    text("h2", "提交仓库与复现")
    text("p", "仓库包含data、scripts、results、videos、evidence与report，附5组裁剪后的4K配对素材（10条）和3条工作/检测视频，按组编号保留来源与对应关系。按README可在本地复现静态实验、动态检测及PDF，输入和提交文件附SHA256校验。" if dynamic else "仓库包含data、scripts、results、evidence与report。按README在Python 3.12环境安装固定依赖，可独立复现检测、审计和PDF；不需云服务器。4张输入JPEG与提交文件均附SHA256校验。坐标和计数应一致，计时与字体渲染可随环境变化。")
    text("h2", "参考文献")
    for ref in [
        "[1] OpenCV 4.11.0. Harris Corner Detection.\nhttps://docs.opencv.org/4.11.0/dc/d0d/tutorial_py_features_harris.html",
        "[2] OpenCV 4.11.0. Introduction to SIFT.\nhttps://docs.opencv.org/4.11.0/da/df5/tutorial_py_sift_intro.html",
        "[3] Lowe DG. Distinctive Image Features from Scale-Invariant Keypoints. IJCV 60(2):91–110, 2004. https://www.cs.ubc.ca/~lowe/papers/ijcv04.pdf",
        "[4] Mikolajczyk K, et al. A Comparison of Affine Region Detectors. IJCV 65(1/2):43–72, 2005. 官方评估说明：https://www.robots.ox.ac.uk/~vgg/research/affine/evaluation.html",
        "[5] Harris C, Stephens M. A Combined Corner and Edge Detector. Alvey Vision Conference, 1988. doi:10.5244/C.2.23. https://www.bmva-archive.org.uk/bmvc/1988/avc-88-023.html",
    ]: text("reference", ref)

    report = {"title": "足球双视角画面的Harris与SIFT特征检测", "subtitle": "计算机视觉第8周实验报告｜图像采集、受控生成与处理", "date": "采集日期：2026年10月19日｜实验环境：Python 3.12 / OpenCV 4.11.0 / CPU", "blocks": blocks, "provenance": {"metrics": relative(results / "metrics.json"), "run_timestamp_utc": metrics["run_timestamp_utc"], "inputs_sha256": {im["image_id"]: im["sha256"] for im in metrics["images"]}, "all_report_values_derived_from_records": True}}
    if dynamic:
        report["provenance"]["dynamic_manifest"] = relative(dynamic / "manifest.json")
    report["provenance"]["revision_scope"] = "Restore process-and-result-first lab report; add abstract, keywords and formal references; measured data unchanged"
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"report_data": str(output), "assets": str(assets), "blocks": len(blocks), "intended_pages": 9 if dynamic else 8}, ensure_ascii=False))


if __name__ == "__main__":
    main()

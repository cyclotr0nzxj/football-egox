#!/usr/bin/env python3
"""Assemble the Week 8 report from saved measurements; does not author a PDF."""
from pathlib import Path
import json
from PIL import Image, ImageOps, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'output/week08'
FONT = '/System/Library/Fonts/Supplemental/Arial Unicode.ttf'


def read(path):
    return json.loads((ROOT / path).read_text())


def assets():
    dest = OUT / 'report_assets'
    dest.mkdir(parents=True, exist_ok=True)
    font = ImageFont.truetype(FONT, 24)
    canvas = Image.new('RGB', (1280, 490), '#eef3f6')
    draw = ImageDraw.Draw(canvas)
    for i in range(2):
        path = ROOT / f'验证视频/拍摄信息/场景2机位{i+1}.JPG'
        with Image.open(path) as source:
            im = ImageOps.contain(ImageOps.exif_transpose(source).convert('RGB'), (630, 440))
        x = i * 640 + (640-im.width)//2
        canvas.paste(im, (x, 45 + (440-im.height)//2))
        draw.text((i*640+12, 8), f'现场场景2 / 机位记录照片{i+1}', font=font, fill='#17212d')
    canvas.save(dest / 'capture_setup.jpg', quality=90)
    canvas = Image.new('RGB', (1680, 994), '#10171e')
    draw = ImageDraw.Draw(canvas)
    labels = [('EXO 官方第三人称输入', 10), ('EGO 官方实拍参考', 794), ('EgoX 生成结果', 1242)]
    for label, x in labels:
        draw.text((x, 8), label, font=font, fill='white')
    for i, n in enumerate([5, 35]):
        y = 42 + i*476
        draw.text((12, y), f'官方样例 F{n:02d}（从0编号）', font=font, fill='white')
        with Image.open(ROOT / f'artifacts/frames/{n:03d}.png') as im:
            assert im.size == (1680, 448)
            canvas.paste(im.convert('RGB'), (0, y+28))
    canvas.save(dest / 'official_generation.jpg', quality=94)
    with Image.open(OUT / 'frames/scene04/exo/0000.jpg') as a, Image.open(OUT / 'frames/scene04/ego/0000.jpg') as b:
        canvas = Image.new('RGB', (1920, 584), '#10171e')
        draw = ImageDraw.Draw(canvas)
        draw.text((12,8), '配对04 EXO / F0', font=font, fill='white')
        draw.text((972,8), '配对04 EGO / F0', font=font, fill='white')
        canvas.paste(a,(0,44)); canvas.paste(b,(960,44))
        canvas.save(dest/'pair04_context.jpg', quality=90)


def main():
    assets()
    seg = read('output/week08/results/image_processing/metrics.json')
    reg = read('output/week08/results/registration/metrics.json')
    gen = read('artifacts/evaluation.json')
    trim = read('records/user_ego_trim_20261019/trim_manifest.json')
    c, t = reg['controlled'], reg['temporal']
    grid = c['independent_grid_transfer_error']
    b = []
    def p(text): b.append(dict(type='p', text=text))
    def h(text): b.append(dict(type='h1', text=text))
    def h2(text): b.append(dict(type='h2', text=text))
    def table(headers, rows, widths=None):
        b.append(dict(type='table', headers=headers, rows=rows, widths=widths or [1]*len(headers)))
    def image(path, caption, height=300):
        b.append(dict(type='image', path='output/week08/'+path, caption=caption, width=490, max_height=height))
    def page(): b.append(dict(type='pagebreak'))

    h('摘要与实验目标')
    p('本实验以足球传接球为场景，建立从双机采集、时间同步、关键帧展示到传统视觉算法处理的完整流程。使用 Canon R8 固定第三人称录像和 DJI Nano 实拍第一人称录像，整理5组同步片段，选配对04完成灰度化、滤波、边缘检测、足球颜色分割、ORB特征检测与匹配、单应性配准。生成部分复用已实际完成的 EgoX 官方足球样例推理，不训练或微调。')
    table(['课程实践','本报告完成内容','证据位置'],[
        ['图像采集','设备设置、现场照片、5组双视角短片','第1、2节'],
        ['图像处理 / 分割','灰度、Gaussian、Canny、HSV与形状规则','第4、5节'],
        ['特征检测 / 匹配','ORB、Hamming近邻匹配、RANSAC内点','第6、7节'],
        ['图像配准','已知透视变换及真实同机时序实验','第6、7节'],
        ['图像生成','官方权重、官方样例的真实已有结果','第8节'],
    ],[1,2.8,1])
    h2('主要结果')
    table(['项目','结果','解释范围'],[
        ['同步素材','5组 / 1571帧每视角','共同片段累计62.847秒'],
        ['足球分割',f"平均IoU {seg['mean_iou_raw']:.3f} → {seg['mean_iou_refined']:.3f}",'6帧粗参考，包含2个失败'],
        ['受控配准',f"独立网格均误差 {grid['mean_px']:.3f} px",'源图固定187点，误差为目标图像素'],
        ['官方生成',f"PSNR {gen['mean_psnr_rgb_db']:.2f} dB / SSIM {gen['mean_ssim_rgb']:.4f}",'官方单样例49帧，非自采结果'],
    ],[1,2,2.2])
    p('本报告将时间同步、二维配准和视角生成分别验证。自采素材的深度、相机几何、逐帧目标轨迹和先验尚未生成，因此不声称已完成自采视频的 EgoX 转换。')

    page(); h('1  图像采集与拍摄记录')
    p('2026年10月19日拍摄小范围足球动作。R8在场边支架上记录第三人称，Nano随主角记录第一人称。现场照片可见设备位于帽檐/头部附近，具体光心位置与安装旋转未测定。主角灰上衣、深色裤、红鞋，足球具有浅色、黄色与紫色纹样。')
    table(['记录项','已核对的内容','范围与缺项'],[
        ['Canon R8','3840×2160；25.00fps；标准IPB','来自R8格式照片，并核对视频'],
        ['R8设置照片','24mm；1/60s；Tv；SERVO AF；曝光补偿0','未显示光圈/ISO，不证明每条都恒定'],
        ['DJI Nano','4K；16:9；25fps','未记录快门、ISO、视场与防抖模式'],
        ['主实验工作图','960×540，保持16:9','原视频四分之一宽高，不覆盖视频'],
        ['现场场景2','机位高度1.23m；至最近盘6.3m；盘间距0.38m','记录未定义光心基准和完整世界坐标'],
    ],[1.1,2,2.4])
    image('report_assets/capture_setup.jpg','图1 现场场景2的两张机位记录照片。配对04与该场景的对应关系来自画面比较，属于视觉推断。',210)
    p('记录照片由iPhone拍摄，其EXIF中的5.7mm是记录手机镜头参数，不能代替R8或Nano标定。焦段、机位高度和单个距离也不足以直接得到相机内参K、畸变系数和外参R、t。现场场景2的盘近似共线，不能仅靠它们解出完整平面单应矩阵。')

    page(); h('2  双视角同步与裁剪')
    p('设备显示日期时间约差18分钟，不能据此直接确定帧级偏移。基础同步先由音频定位拍掌，再逐帧检查手掌接触画面；视觉同步不确定度约±1帧，即25fps下约40ms。随后以用户进一步裁剪的EGO为准，逐个HEVC压缩包计算SHA-256，与基础对齐EGO匹配，恢复唯一的恒定起点，再将同一时间区间转移至EXO。')
    table(['配对','EXO文件','基础对齐版起点(s)','可见帧/视角','视频时长(s)'],[
        [x['id'],Path(x['output_exo']).name,f"{x['start_seconds']:.2f}",str(x['visible_frame_count']),f"{x['duration_seconds']:.6f}"] for x in trim['pairs']
    ],[.55,1.35,1.3,1.2,1.3])
    p('10条裁剪视频均已完整解码。两路可见帧的显示时间戳逐一相同，共1571帧/视角；额外裁剪没有消除原拍掌同步的±1帧误差。MP4可能含隐藏预卷包，文件级帧数不一定等于可见帧数；最后一帧显示时长保留用户裁剪终点，因此不强制用“帧数/25”替换视频时长。EXO视频载荷直接复制，音频重新裁剪编码。')
    image('report_assets/pair04_context.jpg','图2 主实验配对04的F0全幅对照。两幅是时间对应画面，尚未进行跨视角空间配准。',170)
    p('配对04为MVI_5693.MP4及对应Nano片段，共311个可见帧、12.443984秒。以下从短片首个可见帧以0编号，t=i/25；工作图为960×540。短片Fi对应基础对齐版F(99+i)。原始未裁剪备份当前不在项目内，本实验直接读取现存裁剪视频，旧原始帧坐标仅作历史记录。')

    page(); h('3  触球附近关键帧的双视角展示')
    p('通过前后邻帧目视检查，选取脚与球接近、接触或离开过程中的4个代表帧。选帧误差按约±2帧记录；它们是触球附近的展示帧，不是自动触球检测结果，也不是精确的物理接触真值。图中均为局部裁切，保持比例并留黑边。')
    table(['可见帧','时间(s)','用途'],[['48','1.92','脚球接近/处理球过程'],['150','6.00','脚边控球过渡'],['270','10.80','移动中的脚球接近'],['292','11.68','变向处理球附近']],[1,1,3])
    image('results/keyframes/manual_contacts_1.png','图3a F48、F150。左为R8局部，右为Nano局部；使用相同短片帧编号。',215)
    image('results/keyframes/manual_contacts_2.png','图3b F270、F292。两视角提供动作上下文，但可见范围和遮挡不同。',215)

    page(); h('4  图像处理与分割方法')
    p('在EGO F0上展示基本处理链：RGB转灰度，5×5 Gaussian滤波（sigma=1.2），再用Canny阈值80/160检测边缘。滤波用于抑制细碎纹理，Canny用于观察球、鞋与草地的边界；没有无噪声参考图，因此不报告降噪精度。')
    image('results/image_processing/processing_overview.png','图4 同一局部的原图、灰度图、Gaussian结果和Canny边缘，未改变图像几何。',115)
    h2('4.1  足球颜色与形状规则')
    p('采用OpenCV HSV颜色空间，H范围为0到179，S/V为0到255[1]。固定ROI=(250,250)-(750,540)，坐标基于960×540工作图。先取以下三类颜色的并集，再在ROI内处理。参数根据F0试验帧及原图目视观察确定，保存后统一应用于6帧，不按各帧结果分别调参。')
    table(['颜色候选','下界(H,S,V)','上界(H,S,V)'],[['亮色低饱和','(0,0,135)','(179,90,255)'],['黄色','(15,45,160)','(40,255,255)'],['紫色','(110,45,75)','(165,255,255)']],[1.4,1.8,1.8])
    p('候选掩膜先做5×5闭运算2次和3×3开运算，再提取外轮廓并取凸包。保留面积80至15000像素、圆度4*pi*A/P²≥0.45、宽高比0.55至1.60的成分，选择A×圆度²最大的成分并填充。该方法结合颜色、位置和形状，属于规则分割，不是训练的语义分割模型。')
    h2('4.2  评价设计')
    p('以6个有意选择的不同状态帧作小样本教学评估。粗多边形由编码助手依据未处理原图目视绘制，尚未经独立人工复核，不是精细像素真值；不能把这些分数当作通用检测准确率。IoU=交集面积/并集面积；先逐帧计算，再取6帧算术平均，包含ROI外和误检失败。')

    page(); h('5  分割结果与失败案例')
    table(['帧','原始HSV IoU','规则处理后IoU','观察'],[
        [str(x['frame']),f"{x['raw_hsv']['iou']:.3f}",f"{x['refined']['iou']:.3f}", '误选非球成分' if x['frame']==170 else '球位于ROI外' if x['frame']==130 else '足球主体基本保留'] for x in seg['frames']
    ]+[['六帧均值',f"{seg['mean_iou_raw']:.3f}",f"{seg['mean_iou_refined']:.3f}",'全部帧纳入统计']],[.6,1.1,1.2,2.1])
    image('results/image_processing/segmentation_overview.png','图5 F10成功、F170失败、F280成功。左为原图与粗参考轮廓，中为HSV候选，右为最终掩膜叠加。',270)
    p('平均IoU从0.451升至0.619，来自颜色、形态学、凸包、几何过滤与成分选择的整体作用，不能只归因于某一步。6帧中4帧IoU超过0.5；F170被草地亮区、手部等相似候选干扰，成分评分选错；F130的球位于ROI上方，直接漏检。说明固定ROI和简单颜色规则只适合有限场景。')
    image('results/image_processing/failure_outside_roi.png','图6 F130的ROI覆盖失败：球的粗参考位于矩形上方，输出为空，IoU=0。',100)

    page(); h('6  特征检测、匹配与受控图像配准')
    p('使用EXO F50作为源图，先施加已保存的透视矩阵H_true制作目标图。采用ORB提取二进制描述子，BF Hamming的KNN(k=2)匹配，距离比值小于0.75时保留[2]。RANSAC估计H_est，重投影阈值3像素、最多5000次迭代、置信度0.995[3]。随机种子42，OpenCV单线程；ORB最多5000点、scaleFactor=1.2、8层、FAST阈值12。')
    table(['受控实验','结果'],[
        ['关键点（源/目标）',f"{c['source_keypoints']} / {c['target_keypoints']}"],
        ['比值筛选匹配 / RANSAC内点',f"{c['ratio_test_match_count']} / {c['ransac_inlier_count']} ({c['ransac_inlier_ratio']*100:.2f}%)"],
        ['独立187点误差：均值 / 中位数',f"{grid['mean_px']:.3f} / {grid['median_px']:.3f} px"],
        ['独立187点误差：95%分位 / 最大值',f"{grid['p95_px']:.3f} / {grid['maximum_px']:.3f} px"],
    ],[2.5,2])
    image('results/registration/controlled/ransac_inliers.png','图7 RANSAC保留的特征对应。连线表示内点，内点比例本身不等于独立配准精度。',145)
    image('results/registration/controlled/independent_grid.png','图8 独立固定网格的投影对照：黄色为H_true，紫色为H_est，未放大误差。',215)
    p('精度在未用于拟合的17×11固定网格上评价：e_j=||project(H_est,p_j)-project(H_true,p_j)||₂，单位是960×540目标图像素。两个矩阵均为源到目标，回配准目标图时用H_est的逆。结果只验证已知二维透视变换的恢复，不代表真实双机标定、三维恢复或第一视角生成。')

    page(); h('7  真实同机时序配准与适用范围')
    p('进一步对R8同一短片的F50(t=2.00s)与F200(t=8.00s)使用相同匹配/配准参数。该实验含真实主体运动，可观察静态背景的匹配稳定性。背景草地和树木提供纹理，运动人和球不满足同一个静态平面变换。')
    table(['项目','实测结果','含义'],[
        ['比值筛选匹配',str(t['ratio_test_match_count']),'候选对应数'],
        ['RANSAC内点',f"{t['ransac_inlier_count']} / {t['ransac_inlier_ratio']*100:.2f}%",'符合估计模型的对应比例'],
        ['网格相对恒等变换平均位移',f"{t['estimated_grid_displacement_from_identity_diagnostic_only']['mean_px']:.3f} px",'估计变换的位移诊断，非精度'],
    ],[1.7,1.1,2.3])
    image('results/registration/temporal/ransac_inliers.png','图9 真实同机时序匹配。背景对应较多，不能据此判断人物动作已经对齐。',145)
    image('results/registration/temporal/registration_overlay.png','图10 估计变换后的半透明叠加。运动人物产生重影，属于不同时间的内容差异。',230)
    p('这组真实图像没有单应矩阵真值，因此不报告真实配准误差；用于拟合的内点残差也不能替代独立精度。固定支架不保证图像完全无位移，本次未独立查明估计位移的物理原因。R8与Nano存在视差、视场差异和运动遮挡，本报告没有将两者全幅做单一单应变换，也没有把时间拼接冒充空间配准。')

    page(); h('8  生成实验：EgoX官方样例复用')
    p('采用EgoX官方权重，复用项目已于2026年9月11日完成的真实推理[4]。本节使用官方样例cmu_soccer06_6_877_925，完全独立于10月19日R8/Nano自采素材；本次未重新训练、微调或启动新GPU任务。Nano实拍画面没有作为生成首帧、几何先验或提示词来源。')
    table(['项目','实际记录'],[
        ['模型 / 硬件','Wan2.1-I2V-14B-480P-Diffusers + EgoX LoRA / A800 80GB'],
        ['推理配置 / 耗时','seed=42；50步；guidance=5.0；GGA启用；余弦相似度缩放系数3.0 / 19分23秒'],
        ['输出 / 评价',f"49帧；生成分支448×448；PSNR={gen['mean_psnr_rgb_db']:.2f}dB；SSIM={gen['mean_ssim_rgb']:.4f}"],
    ],[1,3.4])
    image('report_assets/official_generation.jpg','图11 官方样例F5与F35：第三人称输入、官方实拍第一人称参考、EgoX生成。不是自采Nano的生成对照。',280)
    p('指标沿用已保存的真实结果评价：官方前49帧按帧索引对应，在原生448×448 RGB全幅（包括黑色鱼眼边界）上逐帧计算后平均；未做后验配准、时间搜索、颜色拟合或边界遮罩。本次核对了相关文件的7项SHA-256。源视频20fps，生成导出30fps，帧索引对应不等于播放秒数相同。指标仅描述这一官方图像平面，原始传感器标定未独立验证。')
    p('生成能保留球场、球门和球员的大致结构，但F35中实拍参考的球已向对方移动，生成球仍偏近；手脚形状、球纹和人物尺度也有偏差。PSNR/SSIM不能单独证明真实动作或视角正确；一个样例不代表一般化性能，也不是完整论文成绩复现。')

    page(); h('9  讨论、边界与后续工作')
    h2('9.1  本次得到的认识')
    p('采集设置决定可用素材的基础。统一横幅4K/25fps方便后期处理，拍掌与画面核验使双视角能围绕同一动作比较；设备日期时间只能辅助找片。真实工作中还要检查显示时间戳和解码可见帧，不能只看文件帧数。')
    p('颜色分割在近景足球颜色清楚时有效，但位置先验过强会漏掉远处的球，相似颜色又会引起误选。应同时展示成功和失败，后续可增加独立人工复核、更多光照与动作样本，并考虑跟踪或学习方法；当前六帧不能支持泛化结论。')
    p('特征匹配能够估计图像间的二维关系，但模型假设必须符合任务。受控实验有已知真值，能够在独立点上评价；真实时序实验没有真值，只能报告对应关系与诊断。第一人称与第三人称的三维视差不能由任意一张单应矩阵普遍消除。')
    h2('9.2  自采EgoX尚缺的输入')
    table(['步骤','当前情况','下一步'],[
        ['同步短片','已完成5组','选择头部可见的连续49帧'],
        ['场景深度 / 相机参数','未运行几何估计','固定相机模式估计深度、K与坐标系'],
        ['目标相机轨迹 / 先验','未完成','依R8几何标注虚拟头部轨迹并渲染'],
        ['投影验收 / 推理','未运行','检查0/24/48帧后使用官方权重'],
    ],[1.4,1.5,2.6])
    p('现场焦段、机位高度和距离可作合理性检查，不能代替逐帧深度、相机内外参、目标轨迹及几何先验。官方Renderer提供ViPE与人工轨迹流程[5]；这些步骤尚未在自采数据上执行。本次现有云连接测试未成功认证，因此没有新建云任务。本作业的生成展示由独立的官方样例承担。')
    p('结论：已完成一个能追溯输入、展示中间处理、量化受控配准并保留失败案例的足球视觉实验。当前成果覆盖课程列出的实践内容；自采第一视角生成作为后续验证目标，完成后才可加入同一素材的三路比较。')

    page(); h('10  复现方法与参考资料')
    p('所有命令从项目根目录执行，使用Python 3.12。依赖版本：OpenCV-headless 4.11.0.86、NumPy 2.3.5、Pillow 12.3.0、ReportLab 4.4.9、pypdf 6.10.0；另需ffmpeg/ffprobe和中文字体。当前机器的完整运行入口见随附README。')
    table(['顺序','命令 / 输出'],[
        ['1 安装环境','python3.12 -m venv .venv-course\n.venv-course/bin/pip install -r output/week08/requirements.txt\nsource .venv-course/bin/activate'],
        ['2 提取工作帧','python scripts/prepare_week08_frames.py\n读取现存裁剪后视频，保存12帧/视角及映射'],
        ['3 图像处理与分割','python scripts/course_image_experiments.py\n输出 results/image_processing 与 results/keyframes'],
        ['4 匹配与配准','python scripts/course_registration.py --source output/week08/frames/scene04/exo/0050.jpg --temporal output/week08/frames/scene04/exo/0200.jpg --output output/week08/results/registration'],
        ['5 组装与导出报告','python scripts/assemble_week08_report.py\npython scripts/build_week08_pdf.py --data output/week08/report_data.json --output output/week08/第8周实验报告.pdf --markdown output/week08/实验报告.md'],
    ],[1,4.5])
    p('分割配置及粗参考：records/course_validation_20261019/segmentation_config_and_annotations.json。同步裁剪依据：records/user_ego_trim_20261019/trim_manifest.json。两项算法的逐帧结果、矩阵及独立网格坐标均保存在output/week08/results/。生成指标来自artifacts/evaluation.json；官方生成的云端复现另见项目REPORT.md，本地课程命令不会启动模型推理。')
    h2('参考资料（访问日期：2026-10-19）')
    for text in [
        '[1] OpenCV 4.11.0. Thresholding Operations using inRange. https://docs.opencv.org/4.11.0/da/d97/tutorial_threshold_inRange.html',
        '[2] OpenCV 4.11.0. Feature Matching. https://docs.opencv.org/4.11.0/dc/dc3/tutorial_py_matcher.html',
        '[3] OpenCV 4.11.0. Feature Matching + Homography to find Objects. https://docs.opencv.org/4.11.0/d1/de0/tutorial_py_feature_homography.html',
        '[4] DAVIAN-Robotics. EgoX: Egocentric Video Generation from a Single Exocentric Video. https://github.com/DAVIAN-Robotics/EgoX',
        '[5] EgoX-EgoPriorRenderer 官方工具与自采流程. https://github.com/kdh8156/EgoX-EgoPriorRenderer',
    ]: p(text)
    p('本报告的数据表和图像来自项目实际文件；程序及报告整理使用编码助手辅助完成。分割粗参考和触球附近选帧的标注方式已分别说明。')
    data = dict(title='足球传接球双视角视觉实验', subtitle='计算机视觉课程 · 前8周阶段实验报告', date='实验与整理日期：2026年10月19日', blocks=b)
    (OUT / 'report_data.json').write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(report_data=str(OUT/'report_data.json'),blocks=len(b),pdf_created=False),ensure_ascii=False))


if __name__ == '__main__': main()

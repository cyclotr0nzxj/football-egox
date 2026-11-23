# 随包裁剪后4K素材

本目录为最终裁剪后的5组EXO/EGO双视角素材，共10条3840×2160、25fps视频，合计396,184,054字节。按现有源文件逐字节复制，不转码；MP4在ZIP中使用ZIP_STORED。

`trimmed_pairs/pair01`至`pair05`各含EXO与EGO。配对04就是本实验使用的裁剪源，只收录一次。所有视频均已经同步并按用户区间裁剪，保留各自音轨。960×540工作副本和动态检测成品另见`data/video_pair04/`与`videos/`。

`manifest.json`与`manifest.csv`记录每条视频的原项目相对路径、包内路径、分辨率、帧率、编码、时长、大小及SHA256。MP4可能带不可见的预滚帧，容器或流的`nb_frames`不等于可见画面数；可见帧数按既有裁剪验收分别为365、350、280、311、265帧，详见`evidence/trim_manifest.json`与`evidence/source_videos_4k_decode_checks.json`。容器时长与可见画面时长也可能有少量差异。

源文件复制前后及包内副本SHA256必须一致。整体ZIP另有CRC和逐文件SHA256校验。这里提供的是裁剪后源素材，未重新编码或补帧。

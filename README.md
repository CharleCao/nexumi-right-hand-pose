# Nexumi 右手彩色多面体位姿

本项目从右手彩色贴片矢量 PDF 和对应 STEP 几何模型建立毫米尺度的三维标记模型，
随后在 EGO 双目 KB 鱼眼视频中检测彩色三角面并估计右手标记的 6DoF 位姿。

当前阶段先完成设计文件解析和贴片—沉面自动匹配。源文件通过命令行传入，不复制、
不修改原始 PDF、STEP 或采集数据。

## 环境

```bash
cd /home/charlie/nexumi/right_hand_pose
source ../.tools/miniforge3/etc/profile.d/conda.sh
conda env create -f environment.yml
conda activate nexumi-right-hand-pose
```

基础环境只包含 STEP 几何解析和单元测试。视频解码、OpenCV 与 GPU 依赖会在几何链路
验证通过后按需加入，避免首阶段安装无关的大型包。

需要处理 EGO 视频时再安装 headless 视觉依赖（不包含 PyTorch/CUDA）：

```bash
python -m pip install -e '.[vision]'
```

## 建立右手标记模型

```bash
nexumi-build-marker \
  --sticker-pdf 'assets/source/右手_V4_0822彩色三角贴片_A4_100pct.pdf' \
  --step 'assets/source/上盖2.stp' \
  --output outputs/right_hand_marker.json
```

仓库的 `assets/source/` 包含本项目使用的原始右手贴纸 PDF 和对应 STEP；不包含左手设计文件。

当前右手设计文件的自动匹配结果：

```text
F0->25  F1->378  F2->54  F3->385  F4->389
F5->400 F6->404  F7->408  F8->412 F9->416
```

三边最大绝对误差为 0.0488 mm。F2/F8/F9 的外边在 STEP 中被拆成多条共线边；F5
另有两个内环。解析器按外轮廓拓扑合并共线点，并保留内环和面积诊断，不依赖上述面号。

生成几何检查图：

```bash
nexumi-render-marker \
  --model outputs/right_hand_marker.json \
  --output outputs/right_hand_marker_views.svg
```

运行测试：

```bash
python -m pytest -q
```

真实双目单帧位姿检查（ROI 只用于首帧初始化，不写死在算法中）：

```bash
python -m nexumi_marker_pose.inspect_pose \
  --calibration /path/to/calibration_camera.yaml \
  --model outputs/right_hand_marker.json \
  --left-video /path/to/camera_left.mp4 \
  --right-video /path/to/camera_right.mp4 \
  --frame 440 --left-roi 850,760,350,300 --right-roi 780,760,360,300 \
  --output-dir outputs/session71_frame440
```

求解器枚举重复颜色的 F0-F9 对应，以双目射线、面心和完整三角角点重投影共同评分，
不依赖真实数据中的面号或画面位置。

`right_hand_marker.json` 当前使用 STEP 原生毫米坐标系。它足以估计稳定的标记位姿；
标记坐标系到手/工具坐标系的固定变换需要依据机械安装关系另行定义。

## 生成连续轨迹

先用上面的单帧命令得到一个已检查的锚点（例如第440帧），再运行：

```bash
source /home/charlie/nexumi/.tools/miniforge3/etc/profile.d/conda.sh
conda activate /home/charlie/nexumi/.tools/miniforge3/envs/nexumi-right-hand-pose
nexumi-track-sequence \
  --session '/home/charlie/nexumi/0829-01-TOYS(71-80)/71' \
  --model /home/charlie/nexumi/right_hand_pose/outputs/right_hand_marker.json \
  --anchor-pose /home/charlie/nexumi/right_hand_pose/outputs/session71_frame440/frame_000440_pose.json \
  --anchor-frame 440 \
  --output-dir /home/charlie/nexumi/right_hand_pose/outputs/session71_trajectory
```

程序自动发现双目 MP4/PTS，按相机时间戳配对，并从锚点向前、向后连续跟踪。
输出 `trajectory.csv`（每帧一行）、`summary.json` 和双目叠加预览
`trajectory_preview.mp4`。`valid=1, interpolated=0` 是实测位姿；短暂漏检会以
`interpolated=1` 标记。

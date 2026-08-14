# 玉米叶部病害检测 Web 系统设计文档

日期：2026-08-14

## 1. 目标

将现有「基于改进 YOLOv8 的田间玉米主要叶部病害实时检测系统」（Python + PyQt6 桌面 GUI）改造成完整的前后端 Web 系统，功能覆盖：

- 图片检测（上传 → 标注图 + 检测框 + 分类统计）
- 批量检测（多文件 → 汇总 + 标注图 zip 下载）
- 摄像头/视频实时检测（WebSocket 推帧）
- 检测参数调节（置信度、IoU、图像尺寸）
- 模型切换（已训练的 best.pt 之间切换）
- 历史记录（图片/批量检测自动保存，可回看/下载）

不重新训练模型，复用已训练权重和共享检测模块。

## 2. 现状与前提（已验证）

- 训练仓库：`F:\CURSOR\corn_disease_detection`（GitHub: DHHD78/real-time-detection-system-for-corn-leaf-diseases）
  - 已抽取共享检测模块 `utils/detector.py`：`find_best_weights` / `load_model` / `extract_detections` / `draw_detections`
  - 已训练权重位于 `runs/train/<实验名>/weights/best.pt`（多个实验，自动扫描）
  - 检测类别（4 类）：玉米锈病、玉米灰斑病、健康玉米、玉米大斑病（`utils/plots.py` 的 `CLASS_NAMES_CN`）
- 运行环境：本机单用户演示；机器有 NVIDIA GPU（torch.cuda 可用时自动用 GPU 推理）
- Python 3.10+ 与 Node/npm 已具备（catdog-web 已验证 FastAPI + Vue 3 环境可运行）

## 3. 技术选型

- 前端：Vue 3 + Vite + Element Plus + axios
- 后端：FastAPI + uvicorn + python-multipart（WebSocket 由 uvicorn 原生支持）
- 推理：复用训练仓库的 `utils/detector.py`，权重只读、不复制
- 前后端分离，开发期通过 Vite 代理 `/api` 与 `/ws` 到 `localhost:8000`

## 4. 项目结构

```
corn-disease-web/
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── main.py            # FastAPI 入口：注册路由、CORS、启动预加载
│   │   ├── config.py          # 训练仓库路径、历史目录、上传限制
│   │   ├── model_manager.py   # 扫描权重、懒加载/缓存/切换模型
│   │   ├── inference.py       # 封装 utils.detector：图片/批量/单帧推理 + 历史落盘
│   │   ├── routes_detect.py   # 图片/批量 REST API
│   │   ├── routes_realtime.py # WebSocket 实时检测
│   │   └── routes_history.py  # 历史记录 API
│   └── requirements.txt
├── frontend/
│   ├── src/
│   │   ├── main.js
│   │   ├── App.vue            # 顶部导航 + 全局参数侧栏
│   │   └── views/
│   │       ├── ImageDetect.vue
│   │       ├── BatchDetect.vue
│   │       ├── Realtime.vue
│   │       └── History.vue
│   ├── index.html
│   ├── vite.config.js         # /api、/ws 代理
│   └── package.json
└── docs/
    └── superpowers/
        ├── specs/             # 本设计文档
        └── plans/             # 实现计划
```

## 5. 后端设计

### 5.1 config.py

- `CORN_PROJECT_ROOT`：训练仓库根目录，默认 `F:\CURSOR\corn_disease_detection`，可用环境变量 `CORN_PROJECT_ROOT` 覆盖
- 权重目录：`<CORN_PROJECT_ROOT>/runs/train`（只读，扫描 `*/weights/best.pt`）
- 历史目录：`<项目根>/runs/web/<时间戳>/`（Web 系统自己落盘，不写训练仓库）
- `MAX_UPLOAD_MB = 20`；`ALLOWED_IMAGE_EXT = {.jpg, .jpeg, .png, .bmp, .tif, .webp}`；`ALLOWED_VIDEO_EXT = {.mp4, .avi, .mov, .mkv}`

### 5.2 model_manager.py

- `scan_models()`：扫描权重，按实验目录修改时间倒序，返回 `[{name, path, mtime, size}]`
- `get_model(name)`：懒加载 + 字典缓存（模型名 → YOLO 实例）；切换时释放旧模型并 `torch.cuda.empty_cache()`
- 启动时预加载最新模型；加载失败仅告警、不阻塞启动，首次请求时重试

### 5.3 inference.py

- `detect_image_bytes(model, img_bytes, conf, iou, img_size)`：cv2 解码 → `model.predict` → `extract_detections` → `draw_detections` → 编码 JPEG；返回标注图字节、detections、分类统计（各类别计数 + 总目标数）
- `save_history(source, params, detections, original, annotated)`：写入 `runs/web/<时间戳>/`，包含 `result.json`、`original.jpg`、`annotated.jpg`、`labels.txt`
- 视频帧推理：单帧复用同一封装；视频文件由服务端 OpenCV 逐帧解码（复用 `detect.py` 的思路，不复制代码）

### 5.4 REST API

- `GET /api/health` → `{status, active_model, models_count, gpu}`
- `GET /api/models` → 模型列表（供前端下拉）
- `POST /api/detect/image`（multipart：`file`、`model`、`conf`、`iou`、`img_size`）→ `{history_id, detections, stats, annotated_url}`
- `POST /api/detect/batch`（multipart：`files[]`、`model`、`conf`、`iou`、`img_size`）→ `{total, results: [{filename, detections, stats}], zip_url}`
- `POST /api/realtime/video`（multipart：`file`）→ `{video_id}`（服务端临时保存，供 WebSocket 推流）
- `GET /api/history` → 历史列表（按时间倒序，缩略图复用标注图的缩小版）
- `GET /api/history/{id}/annotated`、`/original`、`/json`、`/zip` → 回看与下载

### 5.5 WebSocket `/ws/detect`

客户端 → 服务端消息（JSON 或二进制帧）：

- `{type: "config", model, conf, iou, img_size}`：设置本次会话的模型与参数
- `{type: "frame", data: <JPEG 二进制>}`：摄像头帧（浏览器 `getUserMedia` → canvas 抽帧 → 发送）
- `{type: "video", video_id}`：开始服务端视频推流（视频已通过 `POST /api/realtime/video` 上传）
- `{type: "stop"}`：停止当前推流

服务端 → 客户端消息：

- `{type: "result", annotated: <JPEG base64>, detections, stats, fps}`：标注帧
- `{type: "error", message}`：错误
- `{type: "ready"}`：配置成功

帧格式约定：**二进制消息即摄像头 JPEG 帧**（对应 `frame`），JSON 消息为控制与结果消息，两端据此区分。

视频文件流程：前端上传视频 → 返回 `video_id`（临时文件）→ WebSocket 按源 FPS 逐帧推理推流。摄像头走浏览器抽帧（约 15fps、640x480），避免上传整段视频。

视频临时文件在收到 `stop`、连接关闭或服务重启时清理。

### 5.6 并发与性能（单用户）

- 不做任务队列：推理接口在 FastAPI 默认线程池中执行，单用户下天然串行
- 模型缓存为全局单例，GPU 串行推理无竞争
- 目标：640 输入尺寸 + GPU 下实时（20+ FPS 即可满足 15fps 抽帧）

## 6. 前端设计

### 6.1 App.vue

- `el-menu` 顶部导航：图片检测 / 批量检测 / 实时检测 / 历史记录
- 全局参数侧栏：模型下拉（`GET /api/models`）、置信度滑块、IoU、图像尺寸；参数持久化到 localStorage，随请求发送

### 6.2 ImageDetect.vue

- `el-upload` 拖拽上传（格式/大小校验 + 本地预览）
- 提交 → loading → 展示标注图、检测框表格（类别/置信度/坐标）、分类统计
- 自动生成一条历史记录

### 6.3 BatchDetect.vue

- 多文件选择（`el-upload` multiple）
- 同步提交，前端显示 loading；完成后展示逐文件汇总表格 + 下载标注图 zip

### 6.4 Realtime.vue

- 两个来源 Tab：摄像头 / 视频文件
- 摄像头：`getUserMedia` → `<video>` → canvas 抽帧 → WebSocket 发送 → 标注帧显示在 canvas，附带 FPS 与实时统计
- 视频：选择文件 → 上传获取 `video_id` → WebSocket 推流显示
- 停止按钮；断线/错误用 `ElMessage` 提示（不做自动重连）

### 6.5 History.vue

- `el-table` 列表：时间 / 类型 / 模型 / 参数 / 目标数 / 缩略图
- 查看详情（标注图 + 检测框列表）、下载 JSON / zip

### 6.6 vite.config.js

```js
server: {
  proxy: {
    "/api": "http://localhost:8000",
    "/ws": { target: "ws://localhost:8000", ws: true },
  },
}
```

## 7. 数据流

1. 图片：选择/拖拽 → `POST /api/detect/image` → 后端推理 → 存历史 → 返回标注图与检测结果 → 前端渲染
2. 批量：多文件 → `POST /api/detect/batch` → 逐张推理 → 打包 zip → 返回汇总
3. 摄像头：`getUserMedia` → 抽帧 → WebSocket `frame` → 推理 → 返回标注帧 → canvas
4. 视频：上传 → `video_id` → WebSocket `video` → 服务端逐帧推理推流 → canvas
5. 历史：图片/批量检测自动落盘 `runs/web/<时间戳>/`，历史页从 `GET /api/history` 读取

## 8. 错误处理

- 后端：非法格式/超大小 → 400；模型不存在 → 404；推理异常 → 500/503；统一返回中文 `{"detail": "..."}`；WebSocket 异常发送 `{type: "error", message}`
- 前端：非 2xx → `ElMessage.error`；上传/检测中按钮禁用；WebSocket 断线提示并停止

## 9. 测试

- 后端 pytest：health、模型列表、图片检测（stub 模型注入，不依赖真实权重/GPU）、参数校验、批量、历史 API、WebSocket 握手与错误路径
- 前端：`npm run build` 通过；浏览器手动验证四个页面完整流程
- 冒烟：真实权重 + 一张玉米叶图片跑通 `POST /api/detect/image`

## 10. 运行步骤

后端：

```powershell
cd F:\CURSOR\corn-disease-web\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

前端：

```powershell
cd F:\CURSOR\corn-disease-web\frontend
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://localhost:5173`

## 11. 环境依赖

- 后端：fastapi、uvicorn、python-multipart、ultralytics、torch、opencv-python、numpy、Pillow
- 前端：vue、vite、element-plus、axios、@element-plus/icons-vue（npm 管理）
- 说明：后端需能 import 训练仓库的 `utils` 包（通过 `CORN_PROJECT_ROOT` 加入 `sys.path`）；无 GPU 也能运行，仅速度较慢

## 12. 范围外（YAGNI）

- 多用户/登录鉴权、数据库、任务队列、自动重连
- 实时检测不保存历史（历史仅覆盖图片/批量检测）
- 模型上传、在线训练、移动端适配

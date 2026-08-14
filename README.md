# 玉米叶部病害检测 Web 系统

基于改进 YOLOv8 的田间玉米主要叶部病害检测 Web 应用：图片/批量检测、摄像头/视频实时检测、参数调节、模型切换、历史记录。

## 功能

- 图片检测：上传 → 标注图 + 检测框 + 分类统计，自动保存历史
- 批量检测：多文件 → 汇总表格 + 标注图 zip 下载
- 实时检测：摄像头（浏览器抽帧）与视频文件（服务端逐帧推理），WebSocket 推流
- 历史记录：图片/批量检测自动落盘 `runs/web/`，可回看与下载

## 环境要求

- Python 3.10+（后端）
- Node 18+（前端）
- 训练权重位于 `corn_disease_detection/runs/train/*/weights/best.pt`
- 有 NVIDIA GPU 时推理自动使用 GPU（无 GPU 也可运行，速度较慢）

## 启动后端

```powershell
cd F:\CURSOR\corn-disease-web\backend
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

> 训练仓库默认路径 `F:\CURSOR\corn_disease_detection`，可通过环境变量 `CORN_PROJECT_ROOT` 覆盖。

## 启动前端

```powershell
cd F:\CURSOR\corn-disease-web\frontend
npm.cmd install
npm.cmd run dev
```

浏览器打开 `http://localhost:5173`

## 测试

```powershell
cd F:\CURSOR\corn-disease-web\backend
python -m pytest tests/ -v
```

## 目录结构

见 `docs/superpowers/specs/2026-08-14-corn-disease-web-design.md`。

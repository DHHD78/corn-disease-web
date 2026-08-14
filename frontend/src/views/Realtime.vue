<template>
  <div>
    <el-tabs v-model="tab">
      <el-tab-pane label="摄像头" name="camera">
        <el-button type="primary" :disabled="running" @click="startCamera">开始摄像头检测</el-button>
        <el-button :disabled="!running" @click="stop">停止</el-button>
        <p style="color: #909399; font-size: 13px">浏览器将请求摄像头权限，画面以约 15fps 抽帧推送给后端</p>
      </el-tab-pane>
      <el-tab-pane label="视频文件" name="video">
        <el-upload :auto-upload="false" :limit="1" :on-change="onVideo" accept="video/*">
          <el-button>选择视频（mp4/avi/mov/mkv）</el-button>
        </el-upload>
        <div style="margin-top: 12px">
          <el-button type="primary" :disabled="!videoId || running" @click="startVideo">
            开始视频检测
          </el-button>
          <el-button :disabled="!running" @click="stop">停止</el-button>
        </div>
      </el-tab-pane>
    </el-tabs>

    <div style="display: flex; gap: 20px; margin-top: 16px; flex-wrap: wrap">
      <canvas ref="canvas" width="640" height="480" style="border: 1px solid #ddd; background: #000; border-radius: 8px" />
      <div style="min-width: 200px">
        <el-statistic title="FPS" :value="fps" />
        <el-statistic title="目标总数" :value="total" style="margin-top: 16px" />
        <div v-for="(v, k) in classes" :key="k" style="margin-top: 8px">{{ k }}: {{ v }}</div>
      </div>
    </div>

    <video ref="video" style="display: none" playsinline />
  </div>
</template>

<script setup>
import { onBeforeUnmount, ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const tab = ref("camera");
const canvas = ref(null);
const video = ref(null);
const running = ref(false);
const fps = ref(0);
const total = ref(0);
const classes = ref({});
const videoId = ref("");

let ws = null;
let stream = null;
let rafId = null;

function wsUrl() {
  const proto = location.protocol === "https:" ? "wss" : "ws";
  return `${proto}://${location.host}/ws/detect`;
}

function openWs(onOpen) {
  ws = new WebSocket(wsUrl());
  ws.binaryType = "arraybuffer";
  ws.onopen = () => {
    ws.send(
      JSON.stringify({
        type: "config",
        model: params.model,
        conf: params.conf,
        iou: params.iou,
        img_size: params.imgSize,
      })
    );
    onOpen && onOpen();
  };
  ws.onmessage = (ev) => {
    const msg = JSON.parse(ev.data);
    if (msg.type === "result") {
      const img = new Image();
      img.onload = () => {
        const ctx = canvas.value.getContext("2d");
        ctx.drawImage(img, 0, 0, canvas.value.width, canvas.value.height);
      };
      img.src = "data:image/jpeg;base64," + msg.annotated;
      fps.value = msg.fps;
      total.value = msg.stats.total;
      classes.value = msg.stats.classes;
    } else if (msg.type === "error") {
      ElMessage.error(msg.message);
    }
  };
  ws.onclose = () => {
    running.value = false;
    if (rafId) cancelAnimationFrame(rafId);
  };
}

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.value.srcObject = stream;
    await video.value.play();
    openWs(() => {
      running.value = true;
      const sendFrame = () => {
        if (!running.value || !ws || ws.readyState !== WebSocket.OPEN) return;
        const ctx = canvas.value.getContext("2d");
        ctx.drawImage(video.value, 0, 0, 640, 480);
        canvas.value.toBlob((blob) => {
          if (blob && ws.readyState === WebSocket.OPEN) ws.send(blob);
        }, "image/jpeg", 0.7);
        rafId = requestAnimationFrame(sendFrame);
      };
      rafId = requestAnimationFrame(sendFrame);
    });
  } catch (e) {
    ElMessage.error("无法访问摄像头: " + e.message);
  }
}

async function onVideo(f) {
  const form = new FormData();
  form.append("file", f.raw);
  try {
    const { data } = await api.post("/realtime/video", form);
    videoId.value = data.video_id;
    ElMessage.success("视频上传成功，可开始检测");
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "视频上传失败");
  }
}

function startVideo() {
  openWs(() => {
    running.value = true;
    ws.send(JSON.stringify({ type: "video", video_id: videoId.value }));
  });
}

function stop() {
  running.value = false;
  if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify({ type: "stop" }));
  if (ws) ws.close();
  if (stream) {
    stream.getTracks().forEach((t) => t.stop());
    stream = null;
  }
  if (rafId) cancelAnimationFrame(rafId);
}

onBeforeUnmount(stop);
</script>

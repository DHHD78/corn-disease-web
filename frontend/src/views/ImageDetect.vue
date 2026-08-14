<template>
  <div>
    <el-upload
      drag
      :auto-upload="false"
      :limit="1"
      accept="image/*"
      :on-change="onFileChange"
      :on-remove="clear"
    >
      <div style="font-size: 16px">拖拽图片到这里，或点击选择</div>
      <div style="color: #909399; font-size: 13px">支持 jpg / png / bmp / tif / webp，≤20MB</div>
    </el-upload>

    <div v-if="file" style="margin-top: 12px">
      <el-button type="primary" :loading="loading" @click="run">开始检测</el-button>
      <el-button @click="clear">清除</el-button>
    </div>

    <el-alert v-if="error" type="error" :title="error" style="margin-top: 12px" show-icon />

    <div v-if="result" style="display: flex; gap: 20px; margin-top: 16px; flex-wrap: wrap">
      <img :src="result.annotatedUrl" style="max-width: 560px; border: 1px solid #eee; border-radius: 8px" />
      <div style="min-width: 300px; flex: 1">
        <el-descriptions title="检测结果" :column="1" border>
          <el-descriptions-item label="目标总数">{{ result.stats.total }}</el-descriptions-item>
          <el-descriptions-item v-for="(v, k) in result.stats.classes" :key="k" :label="k">
            {{ v }}
          </el-descriptions-item>
        </el-descriptions>
        <el-table :data="result.detections" size="small" max-height="360" style="margin-top: 12px">
          <el-table-column prop="class_name" label="类别" />
          <el-table-column prop="confidence" label="置信度">
            <template #default="{ row }">{{ (row.confidence * 100).toFixed(1) }}%</template>
          </el-table-column>
          <el-table-column label="坐标">
            <template #default="{ row }">{{ row.bbox.map((n) => Math.round(n)).join(", ") }}</template>
          </el-table-column>
        </el-table>
      </div>
    </div>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const file = ref(null);
const loading = ref(false);
const error = ref("");
const result = ref(null);

function onFileChange(f) {
  file.value = f.raw;
  result.value = null;
  error.value = "";
}

function clear() {
  file.value = null;
  result.value = null;
  error.value = "";
}

async function run() {
  if (!file.value) return;
  loading.value = true;
  error.value = "";
  try {
    const form = new FormData();
    form.append("file", file.value);
    form.append("model", params.model);
    form.append("conf", params.conf);
    form.append("iou", params.iou);
    form.append("img_size", params.imgSize);
    const { data } = await api.post("/detect/image", form);
    result.value = { ...data, annotatedUrl: data.annotated_url };
  } catch (e) {
    error.value = e.response?.data?.detail || "检测失败";
    ElMessage.error(error.value);
  } finally {
    loading.value = false;
  }
}
</script>

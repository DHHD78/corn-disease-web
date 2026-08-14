<template>
  <div>
    <el-upload
      drag
      multiple
      :auto-upload="false"
      accept="image/*"
      :on-change="onFiles"
      :on-remove="onFiles"
      :file-list="fileList"
    >
      <div style="font-size: 16px">选择多张图片进行批量检测</div>
      <div style="color: #909399; font-size: 13px">每张 ≤20MB，逐张处理并自动保存历史</div>
    </el-upload>

    <div style="margin-top: 12px">
      <el-button type="primary" :loading="loading" :disabled="!files.length" @click="run">
        批量检测（{{ files.length }} 张）
      </el-button>
      <el-button v-if="zipUrl" type="success" @click="downloadZip">下载标注图 zip</el-button>
    </div>

    <el-table v-if="results.length" :data="results" style="margin-top: 16px">
      <el-table-column prop="filename" label="文件" />
      <el-table-column prop="stats.total" label="目标数" width="100" />
      <el-table-column label="操作" width="200">
        <template #default="{ row }">
          <el-link type="primary" :href="`/api/history/${row.history_id}/annotated`" target="_blank">
            查看标注图
          </el-link>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { ref } from "vue";
import { ElMessage } from "element-plus";
import api from "../api";
import { params } from "../store";

const files = ref([]);
const fileList = ref([]);
const loading = ref(false);
const results = ref([]);
const zipUrl = ref("");

function onFiles(_f, list) {
  fileList.value = list;
  files.value = list.map((i) => i.raw);
}

async function run() {
  loading.value = true;
  try {
    const form = new FormData();
    for (const f of files.value) form.append("files", f);
    form.append("model", params.model);
    form.append("conf", params.conf);
    form.append("iou", params.iou);
    form.append("img_size", params.imgSize);
    const { data } = await api.post("/detect/batch", form);
    results.value = data.results;
    zipUrl.value = data.zip_url;
    ElMessage.success(`处理完成，共 ${data.total} 张`);
  } catch (e) {
    ElMessage.error(e.response?.data?.detail || "批量检测失败");
  } finally {
    loading.value = false;
  }
}

function downloadZip() {
  window.open(zipUrl.value, "_blank");
}
</script>

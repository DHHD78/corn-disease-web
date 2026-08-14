<template>
  <div>
    <el-button style="margin-bottom: 12px" @click="load">刷新</el-button>
    <el-table :data="records" v-loading="loading">
      <el-table-column prop="created_at" label="时间" width="180" />
      <el-table-column prop="source" label="类型" width="100" />
      <el-table-column prop="params.model" label="模型" width="200" />
      <el-table-column prop="stats.total" label="目标数" width="90" />
      <el-table-column label="缩略图" width="130">
        <template #default="{ row }">
          <img :src="row.annotated_url" style="width: 90px; height: 64px; object-fit: cover; border-radius: 4px" />
        </template>
      </el-table-column>
      <el-table-column label="操作" width="280">
        <template #default="{ row }">
          <el-link type="primary" :href="row.annotated_url" target="_blank">标注图</el-link>
          <el-link type="primary" :href="`/api/history/${row.id}/original`" target="_blank" style="margin-left: 12px">
            原图
          </el-link>
          <el-link type="primary" :href="`/api/history/${row.id}/zip`" style="margin-left: 12px">
            下载 zip
          </el-link>
        </template>
      </el-table-column>
    </el-table>
  </div>
</template>

<script setup>
import { onMounted, ref } from "vue";
import api from "../api";

const records = ref([]);
const loading = ref(false);

async function load() {
  loading.value = true;
  try {
    const { data } = await api.get("/history");
    records.value = data;
  } finally {
    loading.value = false;
  }
}

onMounted(load);
</script>

<template>
  <el-container class="app">
    <el-header class="header">
      <span class="title">🌽 玉米叶部病害智能检测系统</span>
      <el-menu mode="horizontal" :default-active="active" @select="active = $event" class="menu">
        <el-menu-item index="image">图片检测</el-menu-item>
        <el-menu-item index="batch">批量检测</el-menu-item>
        <el-menu-item index="realtime">实时检测</el-menu-item>
        <el-menu-item index="history">历史记录</el-menu-item>
      </el-menu>
    </el-header>
    <el-container>
      <el-aside width="260px" class="aside">
        <el-form label-width="90px">
          <el-form-item label="模型">
            <el-select v-model="params.model" style="width: 100%">
              <el-option v-for="m in params.models" :key="m.name" :label="m.name" :value="m.name" />
            </el-select>
          </el-form-item>
          <el-form-item label="置信度">
            <el-slider v-model="params.conf" :min="0.05" :max="0.95" :step="0.05" />
          </el-form-item>
          <el-form-item label="IoU">
            <el-slider v-model="params.iou" :min="0.1" :max="0.9" :step="0.05" />
          </el-form-item>
          <el-form-item label="图像尺寸">
            <el-select v-model="params.imgSize">
              <el-option v-for="s in [416, 512, 640, 768]" :key="s" :label="`${s}`" :value="s" />
            </el-select>
          </el-form-item>
        </el-form>
      </el-aside>
      <el-main>
        <component :is="currentView" />
      </el-main>
    </el-container>
  </el-container>
</template>

<script setup>
import { computed, onMounted, ref, watch } from "vue";
import { loadModels, params, persistParams } from "./store";
import ImageDetect from "./views/ImageDetect.vue";
import BatchDetect from "./views/BatchDetect.vue";
import Realtime from "./views/Realtime.vue";
import History from "./views/History.vue";

const active = ref("image");
const views = { image: ImageDetect, batch: BatchDetect, realtime: Realtime, history: History };
const currentView = computed(() => views[active.value]);

watch(params, persistParams, { deep: true });
onMounted(loadModels);
</script>

<style>
.header { display: flex; align-items: center; gap: 24px; border-bottom: 1px solid #eee; }
.title { font-size: 18px; font-weight: bold; color: #2c3e50; }
.menu { flex: 1; border-bottom: none; }
.aside { border-right: 1px solid #eee; padding: 16px 12px; }
</style>

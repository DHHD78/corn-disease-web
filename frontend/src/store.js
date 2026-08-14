import { reactive } from "vue";
import api from "./api";

const KEY = "corn_disease_params";

function loadSaved() {
  try {
    return JSON.parse(localStorage.getItem(KEY)) || {};
  } catch {
    return {};
  }
}

export const params = reactive({
  model: "",
  conf: 0.25,
  iou: 0.45,
  imgSize: 640,
  models: [],
  ...loadSaved(),
});

export function persistParams() {
  localStorage.setItem(
    KEY,
    JSON.stringify({
      model: params.model,
      conf: params.conf,
      iou: params.iou,
      imgSize: params.imgSize,
    })
  );
}

export async function loadModels() {
  const { data } = await api.get("/models");
  params.models = data;
  if (!params.model && data.length) {
    params.model = data[0].name;
    persistParams();
  }
}

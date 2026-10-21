import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import { defineConfig } from 'vite'

// In dev, Vite serves the app and proxies everything else to a running ComfyUI.
// In prod, ComfyUI serves web/dist itself at /zentrainer/ (see engine/zentrainer/routes.py).
const comfy = process.env.COMFY_URL ?? 'http://127.0.0.1:8188'

export default defineConfig({
  base: '/zentrainer/',
  plugins: [vue()],
  // zenkit-ui is linked from the ZenKit checkout, which has its own vue: one copy only
  resolve: { alias: { '@': fileURLToPath(new URL('./src', import.meta.url)) }, dedupe: ['vue'] },
  build: { target: 'es2022' },
  server: {
    // engine/catalog/families.json is imported from outside web/
    fs: { allow: ['..'] },
    proxy: {
      '/zentrainer/api': comfy,
      '/zenkit': comfy,
      '/object_info': comfy,
      '/settings': comfy,
      '/internal': comfy,
      '/prompt': comfy,
      '/view': comfy,
      '/upload': comfy,
      '/templates': comfy,
      '/ws': { target: comfy.replace(/^http/, 'ws'), ws: true },
    },
  },
})

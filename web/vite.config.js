import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import { fileURLToPath } from 'node:url'
import { dirname, resolve } from 'node:path'

const here = dirname(fileURLToPath(import.meta.url))

// 构建产物落在 web/dist，由 FastAPI 直接托管，无需额外配置
export default defineConfig({
  plugins: [vue()],
  base: './',
  build: {
    outDir: resolve(here, 'dist'),
    emptyOutDir: true,
    assetsDir: 'assets',
    chunkSizeWarningLimit: 1500,
  },
  server: {
    port: 5173,
    proxy: {
      '/api': { target: 'http://127.0.0.1:8600', changeOrigin: true },
    },
  },
})

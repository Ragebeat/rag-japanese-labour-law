import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Прокси на бэкенд, чтобы в коде фронта пути были относительные (/ask).
// Тогда сборка в web/dist работает под FastAPI без единой правки.
export default defineConfig({
  plugins: [react()],
  server: {
    proxy: {
      "/ask": { target: "http://localhost:8000", changeOrigin: true },
      "/health": { target: "http://localhost:8000", changeOrigin: true },
    },
  },
  build: { outDir: "dist" },
});

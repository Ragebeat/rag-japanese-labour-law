import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

// Proxy to the backend, so front-end code uses relative paths (/ask).
// The build in web/dist then runs under FastAPI without a single change.
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

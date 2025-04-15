// webui/vite.config.ts
import { fileURLToPath, URL } from 'node:url'
import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'

// https://vitejs.dev/config/
export default defineConfig({
  plugins: [
    vue(),
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  },
  build: {
    // Output directory relative to the 'webui' folder.
    // This places the built files into 'lollms_server/static/ui'
    // Adjust the path if your structure differs or if you prefer serving
    // directly from 'webui/dist'. We'll assume serving from 'webui/dist'
    // for simplicity in the FastAPI setup below.
    outDir: 'dist',
    assetsDir: 'assets', // Keep assets in an 'assets' subfolder within outDir
    emptyOutDir: true, // Clear the dir before building
  },
  server: {
    port: 5173, // Default Vite port (change if needed)
    strictPort: true, // Fail if port is already in use
    proxy: {
      // Proxy API requests starting with /api/v1 to the FastAPI backend
      '/api/v1': {
        target: 'http://localhost:9600', // Your FastAPI server address
        changeOrigin: true, // Needed for virtual hosted sites
        // secure: false, // Uncomment if backend uses self-signed certificates
        // rewrite: (path) => path.replace(/^\/api\/v1/, '/api/v1'), // Usually not needed if target includes path
        ws: false, // Set to true if you use WebSockets via the API
      }
    }
  }
})
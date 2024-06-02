import { fileURLToPath, URL } from 'node:url'

import { defineConfig } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import VueDevTools from 'vite-plugin-vue-devtools'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'
import eslint from 'vite-plugin-eslint'

// https://vitejs.dev/config/
export default defineConfig({
  css: {
    preprocessorOptions: {
      scss: {
        // additionalData: '@import "@/assets/style/base.scss";'
      }
    }
  },
  plugins: [
    vue(),
    vueJsx(),
    VueDevTools(),
    // element-plus auto import
    AutoImport({
      resolvers: [ElementPlusResolver()]
    }),
    Components({
      resolvers: [ElementPlusResolver()]
    }),
    {
      // do not fail on serve (i.e. local development)
      ...eslint({
        lintOnStart: true,
        include: ['./src'],
        failOnWarning: false,
        failOnError: false
      }),
      apply: 'serve',
      enforce: 'post'
    }
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})

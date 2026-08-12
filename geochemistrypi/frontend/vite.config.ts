import { createHash } from 'node:crypto'
import { execFileSync } from 'node:child_process'
import { fileURLToPath, URL } from 'node:url'

import { defineConfig, type Plugin } from 'vite'
import vue from '@vitejs/plugin-vue'
import vueJsx from '@vitejs/plugin-vue-jsx'
import AutoImport from 'unplugin-auto-import/vite'
import Components from 'unplugin-vue-components/vite'
import { ElementPlusResolver } from 'unplugin-vue-components/resolvers'

const projectRoot = fileURLToPath(new URL('../../', import.meta.url))
const normalizedProjectRoot = projectRoot.replace(/\\/g, '/').replace(/\/$/, '').toLowerCase()
const instanceId =
  process.env.GEOCHEMISTRYPI_ONLINE_INSTANCE_ID ||
  createHash('sha256').update(normalizedProjectRoot, 'utf8').digest('hex').slice(0, 16)

function resolveSourceRevision() {
  if (process.env.GEOCHEMISTRYPI_SOURCE_REVISION) {
    return process.env.GEOCHEMISTRYPI_SOURCE_REVISION
  }
  try {
    return execFileSync('git', ['rev-parse', '--short=12', 'HEAD'], {
      cwd: projectRoot,
      encoding: 'utf8',
      timeout: 2000
    }).trim()
  } catch {
    return 'unknown'
  }
}

const sourceRevision = resolveSourceRevision()
const buildId = process.env.GEOCHEMISTRYPI_BUILD_ID || sourceRevision
const onlineIdentity = {
  service: 'geochemistrypi-online-frontend',
  instance_id: instanceId,
  source_revision: sourceRevision,
  build_id: buildId
}

function onlineIdentityPlugin(): Plugin {
  return {
    name: 'geochemistrypi-online-identity',
    configureServer(server) {
      server.middlewares.use((request, response, next) => {
        if (request.url?.split('?')[0] !== '/__geochemistrypi_instance') {
          next()
          return
        }
        response.statusCode = 200
        response.setHeader('Content-Type', 'application/json; charset=utf-8')
        response.setHeader('Cache-Control', 'no-store')
        response.end(JSON.stringify(onlineIdentity))
      })
    }
  }
}

// https://vitejs.dev/config/
export default defineConfig({
  define: {
    __GEOCHEMISTRYPI_INSTANCE_ID__: JSON.stringify(instanceId),
    __GEOCHEMISTRYPI_SOURCE_REVISION__: JSON.stringify(sourceRevision),
    __GEOCHEMISTRYPI_BUILD_ID__: JSON.stringify(buildId)
  },
  css: {
    preprocessorOptions: {
      scss: {
        // additionalData: '@import "@/assets/style/base.scss";'
      }
    }
  },
  plugins: [
    onlineIdentityPlugin(),
    vue(),
    vueJsx(),
    // element-plus auto import
    AutoImport({
      resolvers: [ElementPlusResolver()]
    }),
    Components({
      resolvers: [ElementPlusResolver()]
    })
  ],
  resolve: {
    alias: {
      '@': fileURLToPath(new URL('./src', import.meta.url))
    }
  }
})

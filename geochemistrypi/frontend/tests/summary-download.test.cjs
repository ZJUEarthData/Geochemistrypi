/**
 * Run from frontend: node --test tests/summary-download.test.cjs
 * Real SFC script/template unit tests with mocked network and downloads.
 * No additional test dependencies; this is not a browser end-to-end test.
 */
const assert = require('node:assert/strict')
const fs = require('node:fs')
const path = require('node:path')
const test = require('node:test')
const vm = require('node:vm')
const ts = require('typescript')
const vue = require('vue')
const { compileTemplate, parse } = require('@vue/compiler-sfc')

const frontend = path.resolve(__dirname, '..')
const read = (relative) => fs.readFileSync(path.join(frontend, relative), 'utf8')
const component = parse(read('src/components/summary-download.vue')).descriptor
const script =
  component.scriptSetup.content +
  '\nexport const testHandles = { downloadSummary, loading, error, summary };'

function evaluate(source, modules, globals = {}) {
  const exports = {}
  const { outputText } = ts.transpileModule(source, {
    compilerOptions: { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2020 }
  })
  vm.runInNewContext(outputText, {
    exports,
    require: (name) => {
      assert.ok(Object.hasOwn(modules, name), `Unexpected import: ${name}`)
      return modules[name]
    },
    Error,
    ...globals
  })
  return exports
}

function deferred() {
  let resolve
  let reject
  const promise = new Promise((yes, no) => {
    resolve = yes
    reject = no
  })
  return { promise, resolve, reject }
}

function result() {
  return {
    artifact: { name: 'summary.zip', download_url: '/artifacts/summary_abc.zip', size_bytes: 100 },
    files: [
      'classification_predictions.csv',
      'classification_report.json',
      'trained_pipeline.joblib'
    ]
  }
}

function harness(t, generate) {
  const requests = []
  const downloads = []
  const unmountCallbacks = []
  const handles = evaluate(
    script,
    {
      vue: { ...vue, onUnmounted: (callback) => unmountCallbacks.push(callback) },
      '@/api/data-mining': {
        generateSummary: (job) => {
          requests.push(job)
          return generate()
        }
      },
      '@/api/online': { artifactUrl: (url) => `https://api.example.test${url}` },
      '@/i18n': { t: (english) => english }
    },
    {
      defineProps: () => ({ jobId: 'job-one' }),
      document: {
        createElement: (tag) => {
          assert.equal(tag, 'a')
          return {
            click() {
              downloads.push({ href: this.href, download: this.download })
            },
            remove() {}
          }
        },
        body: { appendChild() {} }
      }
    }
  ).testHandles
  const unmount = () => unmountCallbacks.splice(0).forEach((callback) => callback())
  t.after(unmount)
  return { ...handles, requests, downloads, unmount }
}

test('API requests a Summary without model parameters or task-tracking state', async () => {
  const calls = []
  const payload = result()
  const api = evaluate(
    read('src/api/data-mining.ts'),
    {
      '@/api/online': { API_BASE_URL: 'https://api.example.test', taskHeaders: () => undefined }
    },
    {
      fetch: async (url, options) => {
        calls.push({ url, options })
        return { ok: true, json: async () => payload }
      }
    }
  )
  assert.equal(await api.generateSummary('job id'), payload)
  assert.equal(calls[0].url, 'https://api.example.test/api/data-mining/jobs/job%20id/summary')
  assert.equal(calls[0].options.method, 'POST')
  assert.equal(calls[0].options.body, undefined)
})

test('API preserves useful backend errors', async () => {
  const api = evaluate(
    read('src/api/data-mining.ts'),
    {
      '@/api/online': { API_BASE_URL: '', taskHeaders: () => undefined }
    },
    {
      fetch: async () => ({
        ok: false,
        json: async () => ({ detail: 'Result files not found for this job' })
      })
    }
  )
  await assert.rejects(api.generateSummary('job-one'), /Result files not found/)
})

test('pending export prevents duplicate requests and successful downloads reuse the URL', async (t) => {
  const pending = deferred()
  const h = harness(t, () => pending.promise)
  const first = h.downloadSummary()
  assert.equal(h.loading.value, true)
  await h.downloadSummary()
  assert.deepEqual(h.requests, ['job-one'])
  pending.resolve(result())
  await first
  assert.equal(h.loading.value, false)
  assert.equal(h.summary.value.files.length, 3)
  await h.downloadSummary()
  assert.equal(h.requests.length, 1)
  assert.equal(h.downloads.length, 2)
  assert.equal(h.downloads[0].download, 'summary.zip')
  assert.equal(h.downloads[0].href, 'https://api.example.test/artifacts/summary_abc.zip')
})

test('failure shows an error and a successful retry clears it', async (t) => {
  let attempt = 0
  const h = harness(t, async () => {
    if (++attempt === 1) throw new Error('Could not prepare Summary. Please retry.')
    return result()
  })
  await h.downloadSummary()
  assert.match(h.error.value, /retry/)
  assert.equal(h.loading.value, false)
  assert.equal(h.summary.value, null)
  assert.equal(h.downloads.length, 0)
  await h.downloadSummary()
  assert.equal(h.error.value, '')
  assert.equal(h.downloads.length, 1)
})

test('non-Error failures also leave a useful retryable state', async (t) => {
  const h = harness(t, async () => {
    throw 'unexpected response'
  })
  await h.downloadSummary()
  assert.match(h.error.value, /failed/)
  assert.equal(h.loading.value, false)
})

test('leaving the result before completion cannot trigger a late download', async (t) => {
  const pending = deferred()
  const h = harness(t, () => pending.promise)
  const request = h.downloadSummary()
  h.unmount()
  pending.resolve(result())
  await request
  assert.equal(h.downloads.length, 0)
  assert.equal(h.summary.value, null)
})

test('all six panels support existing jobs and hide Summary only when artifacts are empty', () => {
  const page = parse(read('src/views/data-mining/data-mining-page.vue')).descriptor
  const nodes = []
  function walk(node) {
    if (node.type === 1 && node.tag === 'SummaryDownload') nodes.push(node)
    for (const child of node.children || []) walk(child)
  }
  walk(page.template.ast)
  const modes = [
    'regressionResult',
    'classificationResult',
    'clusteringResult',
    'dimensionalityReductionResult',
    'anomalyDetectionResult',
    'timeSeriesResult'
  ]
  assert.equal(nodes.length, modes.length)
  for (const [index, node] of nodes.entries()) {
    const compiled = compileTemplate({
      source: node.loc.source,
      filename: 'gate.vue',
      id: 'summary-test'
    })
    assert.deepEqual(compiled.errors, [])
    const { render } = evaluate(compiled.code, {
      vue: { ...vue, resolveComponent: (name) => name }
    })
    const job = { job_id: `job-${index}`, artifacts: [] }
    const context = { [modes[index]]: job }
    assert.equal(render(context, []).type, vue.Comment)
    job.artifacts.push(result().artifact)
    const rendered = render(context, [])
    assert.equal(rendered.type, 'SummaryDownload')
    assert.equal(rendered.props['job-id'], job.job_id)
    assert.equal(rendered.key, job.job_id)
  }
})

test('template retains accessible loading, error and fallback-download states', () => {
  const compiled = compileTemplate({
    source: component.template.content,
    filename: 'summary-download.vue',
    id: 'summary-test'
  })
  assert.deepEqual(compiled.errors, [])
  const { render } = evaluate(compiled.code, { vue: { ...vue, resolveComponent: (name) => name } })
  const context = {
    loading: true,
    error: 'Download failed',
    summary: result(),
    downloadSummary() {},
    t: (english) => english,
    artifactUrl: (url) => url
  }
  const root = render(context, [])
  assert.equal(root.props['aria-busy'], true)
  const heading = root.children[0]
  const button = heading.children[1]
  assert.equal(button.type, 'el-button')
  assert.equal(button.props.disabled, true)
  assert.equal(button.props.loading, true)
  assert.equal(root.children[1].props.role, 'alert')
  assert.equal(root.children[2].children[0].props.download, 'summary.zip')
  assert.match(component.styles[0].content, /@media \(max-width: 680px\)/)
})

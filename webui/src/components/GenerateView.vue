<template>
  <div class="generate">
    <div class="generate-layout">
      <div class="input-panel">
        <div class="card">
          <div class="card-header">Input Configuration</div>
          <div class="card-content">
            <div class="form-group">
              <label>Generation Type:</label>
              <select v-model="request.generation_type" class="select">
                <option value="ttt">Text-to-Text</option>
                <option value="tti">Text-to-Image</option>
                <option value="tts">Text-to-Speech</option>
                <option value="stt">Speech-to-Text</option>
                <option value="ttv">Text-to-Video</option>
                <option value="ttm">Text-to-Music</option>
              </select>
            </div>

            <div class="form-group">
              <label>Personality:</label>
              <select v-model="request.personality" class="select">
                <option value="">Default</option>
                <option v-for="p in store.personalities" :key="p.name" :value="p.name">
                  {{ p.name }}
                </option>
              </select>
            </div>

            <div class="form-group">
              <label>Binding:</label>
              <select v-model="request.binding_name" class="select">
                <option value="">Default</option>
                <option v-for="b in store.activeBindings" :key="b.binding_instance_name" :value="b.binding_instance_name">
                  {{ b.binding_instance_name }}
                </option>
              </select>
            </div>

            <div class="form-group">
              <label>Stream Response:</label>
              <input type="checkbox" v-model="request.stream" class="checkbox">
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">Input Data</div>
          <div class="card-content">
            <div v-for="(input, index) in request.input_data" :key="index" class="input-item">
              <div class="input-header">
                <select v-model="input.type" class="select-small">
                  <option value="text">Text</option>
                  <option value="image">Image</option>
                  <option value="audio">Audio</option>
                  <option value="video">Video</option>
                  <option value="document">Document</option>
                </select>
                <input v-model="input.role" placeholder="Role (e.g., user_prompt)" class="input-small">
                <button @click="removeInput(index)" class="btn-remove">×</button>
              </div>
              <textarea 
                v-if="input.type === 'text'"
                v-model="input.data" 
                placeholder="Enter your text here..."
                class="textarea"
                rows="4"
              ></textarea>
              <div v-else class="file-input">
                <input type="file" @change="handleFileUpload($event, input)" class="file">
                <div v-if="input.data" class="file-preview">
                  File uploaded ({{ input.mime_type }})
                </div>
              </div>
            </div>
            <button @click="addInput" class="btn btn-secondary">Add Input</button>
          </div>
        </div>

        <div class="card">
          <div class="card-header">Parameters</div>
          <div class="card-content">
            <div class="param-grid">
              <div class="form-group">
                <label>Max Tokens:</label>
                <input v-model.number="request.parameters.max_tokens" type="number" class="input">
              </div>
              <div class="form-group">
                <label>Temperature:</label>
                <input v-model.number="request.parameters.temperature" type="number" step="0.1" min="0" max="2" class="input">
              </div>
              <div class="form-group">
                <label>Top P:</label>
                <input v-model.number="request.parameters.top_p" type="number" step="0.1" min="0" max="1" class="input">
              </div>
              <div class="form-group">
                <label>Top K:</label>
                <input v-model.number="request.parameters.top_k" type="number" class="input">
              </div>
            </div>
          </div>
        </div>

        <div class="actions">
          <button @click="generate" class="btn btn-primary" :disabled="!canGenerate || store.loadingStates.generate">
            {{ store.loadingStates.generate ? 'Generating...' : 'Generate' }}
          </button>
          <button @click="clearAll" class="btn btn-secondary">Clear</button>
        </div>
      </div>

      <div class="output-panel">
        <div class="card">
          <div class="card-header">Output</div>
          <div class="card-content">
            <div v-if="store.loadingStates.generate" class="loading">
              Generating response...
            </div>
            <div v-else-if="store.lastError" class="error-message">
              {{ store.lastError }}
            </div>
            <div v-else-if="response" class="response">
              <div v-for="(output, index) in response.output" :key="index" class="output-item">
                <div class="output-header">
                  <span class="output-type">{{ output.type }}</span>
                  <span v-if="output.thoughts" class="thoughts-indicator">💭</span>
                </div>
                <div v-if="output.type === 'text'" class="text-output">
                  {{ output.data }}
                </div>
                <div v-else-if="output.type === 'image'" class="image-output">
                  <img :src="`data:${output.mime_type};base64,${output.data}`" alt="Generated image">
                </div>
                <div v-else class="binary-output">
                  Binary data ({{ output.mime_type }})
                </div>
                <div v-if="output.thoughts" class="thoughts">
                  <strong>Thoughts:</strong> {{ output.thoughts }}
                </div>
              </div>
              <div class="response-meta">
                <div>Execution Time: {{ response.execution_time }}s</div>
                <div>Request ID: {{ response.request_id }}</div>
              </div>
            </div>
            <div v-else class="placeholder">
              Generated content will appear here...
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()

const request = ref({
  input_data: [
    { type: 'text', role: 'user_prompt', data: '', mime_type: null, metadata: {} }
  ],
  personality: '',
  binding_name: '',
  generation_type: 'ttt',
  stream: false,
  parameters: {
    max_tokens: 1024,
    temperature: 0.7,
    top_p: 0.9,
    top_k: 50
  }
})

const response = ref<any>(null)

const canGenerate = computed(() => {
  return request.value.input_data.some(input => input.data.trim() !== '')
})

onMounted(() => {
  if (store.personalities.length === 0) {
    store.loadPersonalities()
  }
  if (store.activeBindings.length === 0) {
    store.loadBindings()
  }
})

function addInput() {
  request.value.input_data.push({
    type: 'text',
    role: '',
    data: '',
    mime_type: null,
    metadata: {}
  })
}

function removeInput(index: number) {
  if (request.value.input_data.length > 1) {
    request.value.input_data.splice(index, 1)
  }
}

function handleFileUpload(event: Event, input: any) {
  const file = (event.target as HTMLInputElement).files?.[0]
  if (file) {
    const reader = new FileReader()
    reader.onload = (e) => {
      const result = e.target?.result as string
      input.data = result.split(',')[1] // Remove data:mime;base64, prefix
      input.mime_type = file.type
    }
    reader.readAsDataURL(file)
  }
}

async function generate() {
  try {
    response.value = null
    const result = await store.generate(request.value)
    response.value = result
  } catch (error) {
    console.error('Generation failed:', error)
  }
}

function clearAll() {
  request.value.input_data = [
    { type: 'text', role: 'user_prompt', data: '', mime_type: null, metadata: {} }
  ]
  response.value = null
  store.lastError = ''
}
</script>

<style scoped>
.generate {
  background: #1a237e;
  color: white;
}

.generate-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  height: calc(100vh - 80px);
}

.input-panel, .output-panel {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.card {
  background: #0d47a1;
  border: 1px solid #1565c0;
  flex: 1;
  display: flex;
  flex-direction: column;
}

.card-header {
  background: #1565c0;
  padding: 12px 16px;
  font-weight: bold;
  border-bottom: 1px solid #1976d2;
}

.card-content {
  padding: 16px;
  flex: 1;
  overflow-y: auto;
}

.form-group {
  margin-bottom: 16px;
}

.form-group label {
  display: block;
  margin-bottom: 4px;
  font-size: 14px;
  font-weight: bold;
}

.select, .input, .textarea {
  width: 100%;
  background: #1a237e;
  border: 1px solid #1565c0;
  color: white;
  padding: 8px 12px;
  font-size: 14px;
}

.select-small, .input-small {
  background: #1a237e;
  border: 1px solid #1565c0;
  color: white;
  padding: 4px 8px;
  font-size: 12px;
}

.textarea {
  resize: vertical;
  min-height: 80px;
}

.checkbox {
  width: auto;
}

.input-item {
  border: 1px solid #1565c0;
  padding: 12px;
  margin-bottom: 12px;
}

.input-header {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
  align-items: center;
}

.btn-remove {
  background: #f44336;
  border: none;
  color: white;
  width: 24px;
  height: 24px;
  cursor: pointer;
  font-size: 16px;
  line-height: 1;
}

.param-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.btn {
  background: #1976d2;
  border: none;
  color: white;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 14px;
  margin-right: 8px;
}

.btn:hover:not(:disabled) {
  background: #1565c0;
}

.btn:disabled {
  background: #424242;
  cursor: not-allowed;
}

.btn-secondary {
  background: #424242;
}

.actions {
  padding: 16px;
  border-top: 1px solid #1565c0;
}

.loading, .placeholder {
  text-align: center;
  color: #bbbbbb;
  padding: 32px;
}

.error-message {
  color: #f44336;
  padding: 16px;
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid #f44336;
}

.output-item {
  border: 1px solid #1565c0;
  padding: 16px;
  margin-bottom: 16px;
}

.output-header {
  display: flex;
  justify-content: space-between;
  margin-bottom: 8px;
  font-weight: bold;
}

.output-type {
  background: #1976d2;
  padding: 2px 8px;
  font-size: 12px;
}

.text-output {
  white-space: pre-wrap;
  line-height: 1.5;
}

.image-output img {
  max-width: 100%;
  height: auto;
}

.thoughts {
  margin-top: 8px;
  padding: 8px;
  background: rgba(76, 175, 80, 0.1);
  border: 1px solid #4caf50;
  font-size: 12px;
}

.response-meta {
  margin-top: 16px;
  padding-top: 16px;
  border-top: 1px solid #1565c0;
  font-size: 12px;
  color: #bbbbbb;
}

.file-input {
  border: 2px dashed #1565c0;
  padding: 16px;
  text-align: center;
}

.file {
  width: 100%;
}

.file-preview {
  margin-top: 8px;
  color: #4caf50;
  font-size: 12px;
}
</style>

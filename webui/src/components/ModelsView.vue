<template>
  <div class="models">
    <div class="models-layout">
      <div class="models-list">
        <div class="card">
          <div class="card-header">
            Discovered Models
            <button @click="store.loadDiscoveredModels()" class="btn-refresh">↻</button>
          </div>
          <div class="card-content">
            <div v-if="store.loadingStates.models" class="loading">Loading models...</div>
            <div v-else-if="Object.keys(store.discoveredModels).length === 0" class="empty">
              No models found in configured folders
            </div>
            <div v-else class="models-categories">
              <div v-for="(models, category) in store.discoveredModels" :key="String(category)" class="category-section">
                <div class="category-header" @click="toggleCategory(String(category))">
                  <span class="category-name">{{ String(category).toUpperCase() }}</span>
                  <span class="category-count">({{ models.length }})</span>
                  <span class="category-toggle">{{ expandedCategories[String(category)] ? '▼' : '▶' }}</span>
                </div>
                <div v-if="expandedCategories[String(category)]" class="category-models">
                  <div
                    v-for="model in models"
                    :key="model"
                    @click="selectModel(String(category), model)"
                    :class="{ active: selectedModel?.category === String(category) && selectedModel?.name === model }"
                    class="model-item"
                  >
                    <div class="model-name">{{ model }}</div>
                    <div class="model-category">{{ category }}</div>
                  </div>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="model-details">
        <div class="card">
          <div class="card-header">
            {{ selectedModel ? 'Model Details' : 'Select a Model' }}
          </div>
          <div class="card-content">
            <div v-if="!selectedModel" class="placeholder">
              Select a model from the list to view details
            </div>
            <div v-else class="details">
              <div class="detail-section">
                <h3>Basic Information</h3>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="label">Name:</span>
                    <span class="value">{{ selectedModel.name }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Category:</span>
                    <span class="value">{{ selectedModel.category }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">File Extension:</span>
                    <span class="value">{{ getFileExtension(selectedModel.name) }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Estimated Type:</span>
                    <span class="value">{{ getModelType(selectedModel.category) }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Compatible Bindings</h3>
                <div class="compatible-bindings">
                  <div v-for="binding in getCompatibleBindings(selectedModel.category)" :key="binding" class="binding-tag">
                    {{ binding }}
                  </div>
                  <div v-if="getCompatibleBindings(selectedModel.category).length === 0" class="no-bindings">
                    No specific bindings identified for this model type
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Usage Information</h3>
                <div class="usage-info">
                  <div class="usage-item">
                    <strong>Model Path:</strong>
                    <code>{{ getModelPath(selectedModel.category, selectedModel.name) }}</code>
                  </div>
                  <div class="usage-item">
                    <strong>Typical Use Cases:</strong>
                    <div class="use-cases">
                      <span v-for="useCase in getUseCases(selectedModel.category)" :key="useCase" class="use-case">
                        {{ useCase }}
                      </span>
                    </div>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Model Analysis</h3>
                <div class="analysis-section">
                  <div class="analysis-item">
                    <strong>Estimated Size:</strong>
                    <span>{{ estimateModelSize(selectedModel.name) }}</span>
                  </div>
                  <div class="analysis-item">
                    <strong>Quantization:</strong>
                    <span>{{ detectQuantization(selectedModel.name) }}</span>
                  </div>
                  <div class="analysis-item">
                    <strong>Model Family:</strong>
                    <span>{{ detectModelFamily(selectedModel.name) }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Quick Actions</h3>
                <div class="actions">
                  <button @click="copyModelName" class="btn btn-secondary">
                    Copy Model Name
                  </button>
                  <button @click="copyModelPath" class="btn btn-secondary">
                    Copy Full Path
                  </button>
                  <button @click="testWithBinding" class="btn btn-primary" :disabled="!hasCompatibleBinding">
                    Test with Binding
                  </button>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()
const selectedModel = ref<any>(null)
const expandedCategories = reactive<Record<string, boolean>>({})

const hasCompatibleBinding = computed(() => {
  if (!selectedModel.value) return false
  return getCompatibleBindings(selectedModel.value.category).length > 0
})

onMounted(() => {
  if (Object.keys(store.discoveredModels).length === 0) {
    store.loadDiscoveredModels()
  }
  // Expand all categories by default
  Object.keys(store.discoveredModels).forEach(category => {
    expandedCategories[category] = true
  })
})

function toggleCategory(category: string) {
  expandedCategories[category] = !expandedCategories[category]
}

function selectModel(category: string, name: string) {
  selectedModel.value = { category, name }
}

function getFileExtension(filename: string): string {
  const ext = filename.split('.').pop()
  return ext ? `.${ext}` : 'Unknown'
}

function getModelType(category: string): string {
  const typeMap: Record<string, string> = {
    'gguf': 'Text Generation (GGUF)',
    'diffusers_models': 'Image Generation (Diffusers)',
    'tts': 'Text-to-Speech',
    'stt': 'Speech-to-Text',
    'ttv': 'Text-to-Video',
    'ttm': 'Text-to-Music',
    'safetensors': 'Various (SafeTensors)',
    'pytorch': 'PyTorch Model',
    'onnx': 'ONNX Model'
  }
  return typeMap[category] || 'Unknown Type'
}

function getCompatibleBindings(category: string): string[] {
  const bindingMap: Record<string, string[]> = {
    'gguf': ['llamacpp_binding', 'ollama_binding'],
    'diffusers_models': ['diffusers_binding'],
    'safetensors': ['diffusers_binding', 'transformers_binding'],
    'pytorch': ['transformers_binding'],
    'onnx': ['onnx_binding']
  }
  return bindingMap[category] || []
}

function getUseCases(category: string): string[] {
  const useCaseMap: Record<string, string[]> = {
    'gguf': ['Chat', 'Text Completion', 'Code Generation', 'Q&A'],
    'diffusers_models': ['Image Generation', 'Image-to-Image', 'Inpainting'],
    'tts': ['Voice Synthesis', 'Audio Generation'],
    'stt': ['Speech Recognition', 'Audio Transcription'],
    'ttv': ['Video Generation', 'Animation'],
    'ttm': ['Music Generation', 'Audio Synthesis']
  }
  return useCaseMap[category] || ['General AI Tasks']
}

function getModelPath(category: string, name: string): string {
  return `models/${category}/${name}`
}

function estimateModelSize(filename: string): string {
  // Simple heuristic based on filename patterns
  if (filename.includes('7b') || filename.includes('7B')) return '~7B parameters'
  if (filename.includes('13b') || filename.includes('13B')) return '~13B parameters'
  if (filename.includes('30b') || filename.includes('30B')) return '~30B parameters'
  if (filename.includes('70b') || filename.includes('70B')) return '~70B parameters'
  if (filename.includes('small')) return 'Small model'
  if (filename.includes('large')) return 'Large model'
  if (filename.includes('xl')) return 'Extra Large model'
  return 'Size unknown'
}

function detectQuantization(filename: string): string {
  if (filename.includes('q4_k_m') || filename.includes('Q4_K_M')) return 'Q4_K_M (4-bit)'
  if (filename.includes('q4_0') || filename.includes('Q4_0')) return 'Q4_0 (4-bit)'
  if (filename.includes('q5_k_m') || filename.includes('Q5_K_M')) return 'Q5_K_M (5-bit)'
  if (filename.includes('q8_0') || filename.includes('Q8_0')) return 'Q8_0 (8-bit)'
  if (filename.includes('f16') || filename.includes('F16')) return 'F16 (16-bit float)'
  if (filename.includes('f32') || filename.includes('F32')) return 'F32 (32-bit float)'
  return 'Unknown/Full precision'
}

function detectModelFamily(filename: string): string {
  const name = filename.toLowerCase()
  if (name.includes('llama')) return 'LLaMA'
  if (name.includes('mistral')) return 'Mistral'
  if (name.includes('gemma')) return 'Gemma'
  if (name.includes('phi')) return 'Phi'
  if (name.includes('qwen')) return 'Qwen'
  if (name.includes('stable-diffusion') || name.includes('sd')) return 'Stable Diffusion'
  if (name.includes('dall-e') || name.includes('dalle')) return 'DALL-E'
  if (name.includes('whisper')) return 'Whisper'
  return 'Unknown Family'
}

function copyModelName() {
  if (selectedModel.value) {
    navigator.clipboard.writeText(selectedModel.value.name)
  }
}

function copyModelPath() {
  if (selectedModel.value) {
    const path = getModelPath(selectedModel.value.category, selectedModel.value.name)
    navigator.clipboard.writeText(path)
  }
}

function testWithBinding() {
  // This would integrate with the generate view
  console.log('Test with binding functionality would be implemented here')
}
</script>

<style scoped>
.models {
  background: #1a237e;
  color: white;
}

.models-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  height: calc(100vh - 80px);
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
  display: flex;
  justify-content: space-between;
  align-items: center;
}

.btn-refresh {
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}

.card-content {
  padding: 16px;
  flex: 1;
  overflow-y: auto;
}

.category-section {
  margin-bottom: 16px;
}

.category-header {
  background: #1976d2;
  padding: 8px 12px;
  cursor: pointer;
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 8px;
}

.category-name {
  font-weight: bold;
}

.category-count {
  color: #bbbbbb;
  font-size: 12px;
}

.category-toggle {
  font-size: 12px;
}

.model-item {
  padding: 8px 12px;
  border: 1px solid #1565c0;
  margin-bottom: 4px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.model-item:hover {
  background: #1565c0;
}

.model-item.active {
  background: #1976d2;
  border-color: #2196f3;
}

.model-name {
  font-weight: bold;
  margin-bottom: 2px;
}

.model-category {
  font-size: 10px;
  color: #bbbbbb;
}

.placeholder {
  text-align: center;
  color: #bbbbbb;
  padding: 32px;
}

.detail-section {
  margin-bottom: 24px;
}

.detail-section h3 {
  margin-bottom: 12px;
  color: #2196f3;
  border-bottom: 1px solid #1565c0;
  padding-bottom: 4px;
}

.detail-grid {
  display: grid;
  gap: 8px;
}

.detail-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.label {
  font-weight: bold;
  color: #bbbbbb;
}

.value {
  color: white;
}

.compatible-bindings {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.binding-tag {
  background: #1976d2;
  padding: 4px 8px;
  font-size: 12px;
}

.no-bindings {
  color: #bbbbbb;
  font-style: italic;
}

.usage-info {
  background: rgba(25, 118, 210, 0.1);
  border: 1px solid #1976d2;
  padding: 12px;
}

.usage-item {
  margin-bottom: 12px;
}

.usage-item:last-child {
  margin-bottom: 0;
}

.use-cases {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
  margin-top: 4px;
}

.use-case {
  background: #424242;
  padding: 2px 6px;
  font-size: 12px;
}

.analysis-section {
  display: grid;
  gap: 8px;
}

.analysis-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.actions {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.btn {
  background: #1976d2;
  border: none;
  color: white;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 14px;
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

.btn-primary {
  background: #1976d2;
}

code {
  background: #424242;
  padding: 2px 4px;
  font-family: monospace;
  font-size: 12px;
}

.loading, .empty {
  text-align: center;
  color: #bbbbbb;
  padding: 16px;
}
</style>

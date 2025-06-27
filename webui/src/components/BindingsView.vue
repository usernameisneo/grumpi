<template>
  <div class="bindings">
    <div class="bindings-layout">
      <div class="bindings-list">
        <div class="card">
          <div class="card-header">
            Active Bindings
            <button @click="store.loadBindings()" class="btn-refresh">↻</button>
          </div>
          <div class="card-content">
            <div v-if="store.loadingStates.bindings" class="loading">Loading bindings...</div>
            <div v-else-if="store.activeBindings.length === 0" class="empty">
              No active bindings found
            </div>
            <div v-else class="binding-list">
              <div
                v-for="binding in store.activeBindings"
                :key="binding.binding_instance_name"
                @click="selectBinding(binding)"
                :class="{ active: selectedBinding?.binding_instance_name === binding.binding_instance_name }"
                class="binding-item"
              >
                <div class="binding-name">{{ binding.binding_instance_name }}</div>
                <div class="binding-type">{{ binding.type }}</div>
                <div class="binding-status">
                  <span class="status-ok">●</span> Active
                </div>
              </div>
            </div>
          </div>
        </div>

        <div class="card">
          <div class="card-header">Available Binding Types</div>
          <div class="card-content">
            <div v-if="store.bindingTypes.length === 0" class="empty">
              No binding types discovered
            </div>
            <div v-else class="type-list">
              <div
                v-for="type in store.bindingTypes"
                :key="type.type_name"
                class="type-item"
              >
                <div class="type-name">{{ type.display_name || type.type_name }}</div>
                <div class="type-version">v{{ type.version || '1.0' }}</div>
                <div class="type-author">{{ type.author || 'Unknown' }}</div>
                <div class="type-description">{{ type.description || 'No description' }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="binding-details">
        <div class="card">
          <div class="card-header">
            {{ selectedBinding ? 'Binding Details' : 'Select a Binding' }}
          </div>
          <div class="card-content">
            <div v-if="!selectedBinding" class="placeholder">
              Select a binding from the list to view details
            </div>
            <div v-else class="details">
              <div class="detail-section">
                <h3>Configuration</h3>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="label">Instance Name:</span>
                    <span class="value">{{ selectedBinding.binding_instance_name }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Type:</span>
                    <span class="value">{{ selectedBinding.type }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedBinding.default_model">
                    <span class="label">Default Model:</span>
                    <span class="value">{{ selectedBinding.default_model }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedBinding.host">
                    <span class="label">Host:</span>
                    <span class="value">{{ selectedBinding.host }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Available Models</h3>
                <div class="models-section">
                  <button @click="loadModelsForBinding" class="btn btn-secondary" :disabled="loadingModels">
                    {{ loadingModels ? 'Loading...' : 'Load Models' }}
                  </button>
                  <div v-if="bindingModels.length > 0" class="models-list">
                    <div v-for="model in bindingModels" :key="model.name" class="model-item">
                      <div class="model-name">{{ model.name }}</div>
                      <div class="model-info">
                        <span v-if="model.family" class="model-family">{{ model.family }}</span>
                        <span v-if="model.format" class="model-format">{{ model.format }}</span>
                        <span v-if="model.quantization_level" class="model-quant">{{ model.quantization_level }}</span>
                      </div>
                      <div v-if="model.size" class="model-size">
                        {{ formatBytes(model.size) }}
                      </div>
                    </div>
                  </div>
                  <div v-else-if="modelsLoaded" class="empty">
                    No models found for this binding
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Model Information</h3>
                <div class="model-info-section">
                  <button @click="loadModelInfo" class="btn btn-secondary" :disabled="loadingModelInfo">
                    {{ loadingModelInfo ? 'Loading...' : 'Get Model Info' }}
                  </button>
                  <div v-if="modelInfo" class="model-info-display">
                    <div class="detail-grid">
                      <div class="detail-item">
                        <span class="label">Model Name:</span>
                        <span class="value">{{ modelInfo.model_name }}</span>
                      </div>
                      <div class="detail-item" v-if="modelInfo.model_type">
                        <span class="label">Type:</span>
                        <span class="value">{{ modelInfo.model_type }}</span>
                      </div>
                      <div class="detail-item" v-if="modelInfo.context_size">
                        <span class="label">Context Size:</span>
                        <span class="value">{{ modelInfo.context_size.toLocaleString() }} tokens</span>
                      </div>
                      <div class="detail-item" v-if="modelInfo.max_output_tokens">
                        <span class="label">Max Output:</span>
                        <span class="value">{{ modelInfo.max_output_tokens.toLocaleString() }} tokens</span>
                      </div>
                      <div class="detail-item">
                        <span class="label">Vision Support:</span>
                        <span class="value">{{ modelInfo.supports_vision ? 'Yes' : 'No' }}</span>
                      </div>
                      <div class="detail-item">
                        <span class="label">Audio Support:</span>
                        <span class="value">{{ modelInfo.supports_audio ? 'Yes' : 'No' }}</span>
                      </div>
                    </div>
                  </div>
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
import { ref, onMounted } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()
const selectedBinding = ref<any>(null)
const bindingModels = ref<any[]>([])
const modelInfo = ref<any>(null)
const loadingModels = ref(false)
const loadingModelInfo = ref(false)
const modelsLoaded = ref(false)

onMounted(() => {
  if (store.activeBindings.length === 0) {
    store.loadBindings()
  }
})

function selectBinding(binding: any) {
  selectedBinding.value = binding
  bindingModels.value = []
  modelInfo.value = null
  modelsLoaded.value = false
}

async function loadModelsForBinding() {
  if (!selectedBinding.value) return
  
  try {
    loadingModels.value = true
    const result = await store.getAvailableModels(selectedBinding.value.binding_instance_name)
    bindingModels.value = result.models || []
    modelsLoaded.value = true
  } catch (error) {
    console.error('Failed to load models:', error)
  } finally {
    loadingModels.value = false
  }
}

async function loadModelInfo() {
  if (!selectedBinding.value) return
  
  try {
    loadingModelInfo.value = true
    modelInfo.value = await store.getModelInfo(selectedBinding.value.binding_instance_name)
  } catch (error) {
    console.error('Failed to load model info:', error)
  } finally {
    loadingModelInfo.value = false
  }
}

function formatBytes(bytes: number): string {
  if (bytes === 0) return '0 B'
  const k = 1024
  const sizes = ['B', 'KB', 'MB', 'GB', 'TB']
  const i = Math.floor(Math.log(bytes) / Math.log(k))
  return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i]
}
</script>

<style scoped>
.bindings {
  background: #1a237e;
  color: white;
}

.bindings-layout {
  display: grid;
  grid-template-columns: 1fr 1fr;
  gap: 16px;
  height: calc(100vh - 80px);
}

.bindings-list {
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

.binding-item, .type-item {
  padding: 12px;
  border: 1px solid #1565c0;
  margin-bottom: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.binding-item:hover, .type-item:hover {
  background: #1565c0;
}

.binding-item.active {
  background: #1976d2;
  border-color: #2196f3;
}

.binding-name, .type-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.binding-type, .type-version {
  font-size: 12px;
  color: #bbbbbb;
  margin-bottom: 4px;
}

.binding-status {
  font-size: 12px;
  display: flex;
  align-items: center;
  gap: 4px;
}

.status-ok {
  color: #4caf50;
}

.type-author {
  font-size: 12px;
  color: #bbbbbb;
  margin-bottom: 4px;
}

.type-description {
  font-size: 12px;
  color: #cccccc;
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

.models-section, .model-info-section {
  margin-top: 12px;
}

.btn {
  background: #1976d2;
  border: none;
  color: white;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 14px;
  margin-bottom: 12px;
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

.models-list {
  max-height: 200px;
  overflow-y: auto;
}

.model-item {
  padding: 8px;
  border: 1px solid #1565c0;
  margin-bottom: 4px;
}

.model-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.model-info {
  display: flex;
  gap: 8px;
  margin-bottom: 4px;
}

.model-family, .model-format, .model-quant {
  background: #1976d2;
  padding: 2px 6px;
  font-size: 10px;
}

.model-size {
  font-size: 12px;
  color: #bbbbbb;
}

.loading, .empty {
  text-align: center;
  color: #bbbbbb;
  padding: 16px;
}

.model-info-display {
  margin-top: 12px;
  padding: 12px;
  background: rgba(25, 118, 210, 0.1);
  border: 1px solid #1976d2;
}
</style>

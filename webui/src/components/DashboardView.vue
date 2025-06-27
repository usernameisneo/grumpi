<template>
  <div class="dashboard">
    <!-- Connection Manager -->
    <ConnectionManager />

    <!-- Server Information -->
    <div v-if="store.isConnected" class="dashboard-grid">
      <div class="card">
        <div class="card-header">Server Information</div>
        <div class="card-content">
          <div class="info-grid">
            <div class="info-item">
              <span class="label">Status:</span>
              <span :class="store.isConnected ? 'status-ok' : 'status-error'">
                {{ store.connectionState.toUpperCase() }}
              </span>
            </div>
            <div class="info-item" v-if="store.serverInfo">
              <span class="label">Version:</span>
              <span>{{ store.serverInfo.version || 'Unknown' }}</span>
            </div>
            <div class="info-item" v-if="store.serverInfo">
              <span class="label">Authentication:</span>
              <span>{{ store.serverInfo.api_key_required ? 'Required' : 'Not Required' }}</span>
            </div>
            <div class="info-item">
              <span class="label">Available Features:</span>
              <span>{{ store.availableFeatures.size }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Real Data Statistics -->
      <div class="card">
        <div class="card-header">Data Summary</div>
        <div class="card-content">
          <div class="stat-grid">
            <div class="stat-item" :class="{ 'stat-available': store.activeBindings.length > 0 }">
              <div class="stat-value">{{ store.activeBindings.length }}</div>
              <div class="stat-label">Active Bindings</div>
              <div class="stat-status">{{ getDataStatus('bindings') }}</div>
            </div>
            <div class="stat-item" :class="{ 'stat-available': store.personalities.length > 0 }">
              <div class="stat-value">{{ store.personalities.length }}</div>
              <div class="stat-label">Personalities</div>
              <div class="stat-status">{{ getDataStatus('personalities') }}</div>
            </div>
            <div class="stat-item" :class="{ 'stat-available': store.functions.length > 0 }">
              <div class="stat-value">{{ store.functions.length }}</div>
              <div class="stat-label">Functions</div>
              <div class="stat-status">{{ getDataStatus('functions') }}</div>
            </div>
            <div class="stat-item" :class="{ 'stat-available': Object.keys(store.discoveredModels).length > 0 }">
              <div class="stat-value">{{ Object.keys(store.discoveredModels).length }}</div>
              <div class="stat-label">Model Categories</div>
              <div class="stat-status">{{ getDataStatus('models') }}</div>
            </div>
          </div>
        </div>
      </div>

      <!-- Default Bindings Configuration -->
      <div class="card" v-if="hasDefaultBindings">
        <div class="card-header">Default Bindings</div>
        <div class="card-content">
          <div v-if="!Object.keys(store.defaultBindings).length" class="no-data">
            No default bindings configured
          </div>
          <div v-else class="defaults-grid">
            <div v-for="(value, key) in store.defaultBindings" :key="String(key)" class="default-item">
              <span class="default-label">{{ formatBindingLabel(String(key)) }}:</span>
              <span class="default-value">{{ value || 'Not Set' }}</span>
            </div>
          </div>
        </div>
      </div>

      <!-- Active Bindings Status -->
      <div class="card" v-if="store.activeBindings.length > 0">
        <div class="card-header">Active Bindings Status</div>
        <div class="card-content">
          <div class="bindings-list">
            <div v-for="binding in store.activeBindings.slice(0, 5)" :key="binding.binding_instance_name" class="binding-item">
              <div class="binding-name">{{ binding.binding_instance_name }}</div>
              <div class="binding-type">{{ binding.type }}</div>
              <div class="binding-status">
                <span class="status-indicator status-ok"></span>
                Active
              </div>
            </div>
            <div v-if="store.activeBindings.length > 5" class="more-bindings">
              ... and {{ store.activeBindings.length - 5 }} more
            </div>
          </div>
        </div>
      </div>

      <!-- System Status -->
      <div class="card full-width">
        <div class="card-header">System Status</div>
        <div class="card-content">
          <div v-if="store.lastError" class="error-section">
            <div class="error-message">{{ store.lastError }}</div>
          </div>
          <div v-else-if="isLoading" class="loading-section">
            <div class="loading-text">Loading system data...</div>
          </div>
          <div v-else class="status-grid">
            <div class="status-item">
              <span class="status-label">Connection:</span>
              <span class="status-value">{{ store.connectionState }}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Features Available:</span>
              <span class="status-value">{{ Array.from(store.availableFeatures).join(', ') || 'None discovered' }}</span>
            </div>
            <div class="status-item">
              <span class="status-label">Authentication:</span>
              <span class="status-value">{{ store.canMakeAuthenticatedRequests ? 'Ready' : 'Required' }}</span>
            </div>
          </div>
        </div>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { computed } from 'vue'
import { useMainStore } from '../stores/main'
import ConnectionManager from './ConnectionManager.vue'

const store = useMainStore()

const isLoading = computed(() =>
  Object.values(store.loadingStates).some(loading => loading)
)

const hasDefaultBindings = computed(() =>
  store.availableFeatures.has('get_default_bindings')
)

function getDataStatus(dataType: string): string {
  if (store.loadingStates[dataType]) return 'Loading...'

  const featureMap: Record<string, string> = {
    'bindings': 'list_bindings',
    'personalities': 'list_personalities',
    'functions': 'list_functions',
    'models': 'list_models'
  }

  const endpoint = featureMap[dataType]
  if (endpoint && store.availableFeatures.has(endpoint)) {
    return 'Loaded'
  }

  return 'Not Available'
}

function formatBindingLabel(key: string): string {
  const labelMap: Record<string, string> = {
    'ttt_binding': 'Text-to-Text',
    'tti_binding': 'Text-to-Image',
    'tts_binding': 'Text-to-Speech',
    'stt_binding': 'Speech-to-Text',
    'ttv_binding': 'Text-to-Video',
    'ttm_binding': 'Text-to-Music'
  }

  return labelMap[key] || key.replace(/_/g, ' ').replace(/\b\w/g, l => l.toUpperCase())
}
</script>

<style scoped>
.dashboard {
  background: #1a237e;
  color: white;
  padding: 16px;
}

.dashboard-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 16px;
  margin-top: 16px;
}

.card {
  background: #0d47a1;
  border: 1px solid #1565c0;
}

.card.full-width {
  grid-column: 1 / -1;
}

.card-header {
  background: #1565c0;
  padding: 12px 16px;
  font-weight: bold;
  border-bottom: 1px solid #1976d2;
  font-size: 14px;
}

.card-content {
  padding: 16px;
}

.info-grid {
  display: grid;
  gap: 8px;
}

.info-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.label {
  font-weight: bold;
  color: #bbbbbb;
}

.status-ok {
  color: #4caf50;
  font-weight: bold;
}

.status-error {
  color: #f44336;
  font-weight: bold;
}

.stat-grid {
  display: grid;
  grid-template-columns: repeat(2, 1fr);
  gap: 16px;
}

.stat-item {
  text-align: center;
  padding: 12px;
  background: #1a237e;
  border: 1px solid #1565c0;
}

.stat-item.stat-available {
  border-color: #4caf50;
}

.stat-value {
  font-size: 24px;
  font-weight: bold;
  color: #2196f3;
  margin-bottom: 4px;
}

.stat-item.stat-available .stat-value {
  color: #4caf50;
}

.stat-label {
  font-size: 12px;
  color: #bbbbbb;
  margin-bottom: 4px;
}

.stat-status {
  font-size: 10px;
  color: #888888;
}

.defaults-grid {
  display: grid;
  gap: 8px;
}

.default-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.default-label {
  font-weight: bold;
  color: #bbbbbb;
}

.default-value {
  color: white;
}

.no-data {
  text-align: center;
  color: #888888;
  font-style: italic;
  padding: 16px;
}

.bindings-list {
  display: grid;
  gap: 8px;
}

.binding-item {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 8px;
  background: #1a237e;
  border: 1px solid #1565c0;
}

.binding-name {
  font-weight: bold;
  font-size: 14px;
}

.binding-type {
  font-size: 12px;
  color: #bbbbbb;
}

.binding-status {
  display: flex;
  align-items: center;
  gap: 4px;
  font-size: 12px;
}

.status-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.status-indicator.status-ok {
  background: #4caf50;
}

.more-bindings {
  text-align: center;
  color: #888888;
  font-style: italic;
  font-size: 12px;
}

.error-section {
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid #f44336;
  padding: 12px;
}

.error-message {
  color: #f44336;
  font-size: 14px;
}

.loading-section {
  text-align: center;
  padding: 16px;
}

.loading-text {
  color: #bbbbbb;
}

.status-grid {
  display: grid;
  gap: 8px;
}

.status-item {
  display: flex;
  justify-content: space-between;
  padding: 4px 0;
}

.status-label {
  font-weight: bold;
  color: #bbbbbb;
}

.status-value {
  color: white;
  font-size: 12px;
}
</style>

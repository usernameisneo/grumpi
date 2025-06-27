<template>
  <div class="connection-manager">
    <!-- Connection Status Header -->
    <div class="connection-header">
      <div class="server-status">
        <div class="status-indicator" :class="statusClass"></div>
        <div class="status-info">
          <div class="status-text">{{ statusText }}</div>
          <div class="server-info" v-if="store.serverInfo">
            LOLLMS Server {{ store.serverInfo.version }}
          </div>
        </div>
      </div>
      <div class="connection-actions">
        <button @click="reconnect" :disabled="isConnecting" class="btn btn-primary">
          {{ isConnecting ? 'Connecting...' : 'Reconnect' }}
        </button>
      </div>
    </div>

    <!-- API Key Setup (when required) -->
    <div v-if="store.needsApiKey" class="api-key-section">
      <div class="section-header">
        <h3>API Key Required</h3>
        <p>This server requires an API key for access to protected endpoints.</p>
      </div>
      <div class="api-key-form">
        <div class="input-group">
          <input 
            v-model="apiKeyInput" 
            type="password" 
            placeholder="Enter your API key"
            class="api-key-input"
            @keyup.enter="setApiKey"
          />
          <button @click="setApiKey" :disabled="!apiKeyInput.trim()" class="btn btn-primary">
            Set Key
          </button>
        </div>
        <div class="api-key-help">
          <p>The API key should be configured in the server's security settings.</p>
        </div>
      </div>
    </div>

    <!-- Connection Error Display -->
    <div v-if="store.lastError" class="error-section">
      <div class="error-header">
        <h3>Connection Error</h3>
      </div>
      <div class="error-content">
        <div class="error-message">{{ store.lastError }}</div>
        <div class="error-actions">
          <button @click="clearError" class="btn btn-secondary">Dismiss</button>
          <button @click="reconnect" class="btn btn-primary">Retry</button>
        </div>
      </div>
    </div>

    <!-- Feature Discovery Status -->
    <div v-if="store.isConnected && store.canMakeAuthenticatedRequests" class="features-section">
      <div class="section-header">
        <h3>Available Features</h3>
        <p>Discovering server capabilities...</p>
      </div>
      <div class="features-grid">
        <div 
          v-for="feature in featureList" 
          :key="feature.id"
          class="feature-item"
          :class="{ 
            available: store.availableFeatures.has(feature.endpoint),
            loading: store.loadingStates[feature.id],
            error: false
          }"
        >
          <div class="feature-icon">{{ feature.icon }}</div>
          <div class="feature-info">
            <div class="feature-name">{{ feature.name }}</div>
            <div class="feature-status">{{ getFeatureStatus(feature) }}</div>
          </div>
          <div class="feature-action">
            <button 
              v-if="!store.availableFeatures.has(feature.endpoint) && !store.loadingStates[feature.id]"
              @click="testFeature(feature)"
              class="btn btn-small"
            >
              Test
            </button>
          </div>
        </div>
      </div>
    </div>

    <!-- Quick Actions -->
    <div v-if="store.isConnected" class="actions-section">
      <div class="section-header">
        <h3>Quick Actions</h3>
      </div>
      <div class="actions-grid">
        <button @click="loadAllData" :disabled="isLoadingAll" class="action-btn">
          <span class="action-icon">🔄</span>
          <span class="action-text">{{ isLoadingAll ? 'Loading...' : 'Refresh All Data' }}</span>
        </button>
        <button @click="clearAllData" class="action-btn">
          <span class="action-icon">🗑️</span>
          <span class="action-text">Clear Cache</span>
        </button>
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()
const apiKeyInput = ref('')

const featureList = [
  { id: 'bindings', name: 'Bindings', endpoint: 'list_bindings', icon: '🔌' },
  { id: 'personalities', name: 'Personalities', endpoint: 'list_personalities', icon: '👤' },
  { id: 'functions', name: 'Functions', endpoint: 'list_functions', icon: '⚙️' },
  { id: 'models', name: 'Models', endpoint: 'list_models', icon: '🤖' },
  { id: 'generate', name: 'Generation', endpoint: 'generate', icon: '🚀' }
]

const isConnecting = computed(() => store.connectionState === 'connecting')
const isLoadingAll = computed(() => 
  Object.values(store.loadingStates).some(loading => loading)
)

const statusClass = computed(() => {
  switch (store.connectionState) {
    case 'connected': return 'status-connected'
    case 'connecting': return 'status-connecting'
    case 'auth_required': return 'status-auth'
    case 'error': return 'status-error'
    default: return 'status-disconnected'
  }
})

const statusText = computed(() => {
  switch (store.connectionState) {
    case 'connected': return 'Connected'
    case 'connecting': return 'Connecting...'
    case 'auth_required': return 'Authentication Required'
    case 'error': return 'Connection Error'
    default: return 'Disconnected'
  }
})

function getFeatureStatus(feature: any): string {
  if (store.loadingStates[feature.id]) return 'Testing...'
  if (store.availableFeatures.has(feature.endpoint)) return 'Available'
  return 'Unknown'
}

async function reconnect() {
  await store.checkHealth()
  if (store.isConnected && store.canMakeAuthenticatedRequests) {
    await loadAllData()
  }
}

function setApiKey() {
  if (apiKeyInput.value.trim()) {
    store.setApiKey(apiKeyInput.value.trim())
    apiKeyInput.value = ''
    reconnect()
  }
}

function clearError() {
  store.lastError = ''
}

async function testFeature(feature: any) {
  const loadFunction = getLoadFunction(feature.id)
  if (loadFunction) {
    await loadFunction()
  }
}

function getLoadFunction(featureId: string) {
  switch (featureId) {
    case 'bindings': return store.loadBindings
    case 'personalities': return store.loadPersonalities
    case 'functions': return store.loadFunctions
    case 'models': return store.loadDiscoveredModels
    default: return null
  }
}

async function loadAllData() {
  if (!store.canMakeAuthenticatedRequests) return
  
  await Promise.allSettled([
    store.loadBindings(),
    store.loadPersonalities(),
    store.loadFunctions(),
    store.loadDiscoveredModels(),
    store.loadDefaultBindings()
  ])
}

function clearAllData() {
  store.clearData()
}

onMounted(() => {
  store.initialize()
})
</script>

<style scoped>
.connection-manager {
  background: #1a237e;
  color: white;
  padding: 16px;
}

.connection-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: #0d47a1;
  border: 1px solid #1565c0;
  margin-bottom: 16px;
}

.server-status {
  display: flex;
  align-items: center;
  gap: 12px;
}

.status-indicator {
  width: 12px;
  height: 12px;
  border-radius: 50%;
  flex-shrink: 0;
}

.status-connected { background: #4caf50; }
.status-connecting { background: #ff9800; animation: pulse 1s infinite; }
.status-auth { background: #2196f3; }
.status-error { background: #f44336; }
.status-disconnected { background: #666666; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.status-info {
  display: flex;
  flex-direction: column;
  gap: 2px;
}

.status-text {
  font-weight: bold;
  font-size: 14px;
}

.server-info {
  font-size: 12px;
  color: #bbbbbb;
}

.section-header {
  margin-bottom: 16px;
}

.section-header h3 {
  margin: 0 0 4px 0;
  color: #2196f3;
  font-size: 16px;
}

.section-header p {
  margin: 0;
  font-size: 12px;
  color: #bbbbbb;
}

.api-key-section, .error-section, .features-section, .actions-section {
  background: #0d47a1;
  border: 1px solid #1565c0;
  padding: 16px;
  margin-bottom: 16px;
}

.input-group {
  display: flex;
  gap: 8px;
  margin-bottom: 8px;
}

.api-key-input {
  flex: 1;
  background: #1a237e;
  border: 1px solid #1565c0;
  color: white;
  padding: 8px 12px;
  font-size: 14px;
}

.api-key-help {
  font-size: 12px;
  color: #bbbbbb;
}

.error-message {
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid #f44336;
  padding: 12px;
  margin-bottom: 12px;
  color: #f44336;
  font-size: 14px;
}

.error-actions {
  display: flex;
  gap: 8px;
}

.features-grid {
  display: grid;
  gap: 8px;
}

.feature-item {
  display: flex;
  align-items: center;
  gap: 12px;
  padding: 12px;
  background: #1a237e;
  border: 1px solid #1565c0;
}

.feature-item.available {
  border-color: #4caf50;
}

.feature-item.loading {
  border-color: #ff9800;
}

.feature-icon {
  font-size: 20px;
  width: 24px;
  text-align: center;
}

.feature-info {
  flex: 1;
}

.feature-name {
  font-weight: bold;
  font-size: 14px;
}

.feature-status {
  font-size: 12px;
  color: #bbbbbb;
}

.actions-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 12px;
}

.action-btn {
  display: flex;
  align-items: center;
  gap: 8px;
  padding: 12px 16px;
  background: #1976d2;
  border: none;
  color: white;
  cursor: pointer;
  transition: background-color 0.2s;
}

.action-btn:hover:not(:disabled) {
  background: #1565c0;
}

.action-btn:disabled {
  background: #424242;
  cursor: not-allowed;
}

.action-icon {
  font-size: 16px;
}

.btn {
  background: #1976d2;
  border: none;
  color: white;
  padding: 8px 16px;
  cursor: pointer;
  font-size: 14px;
  transition: background-color 0.2s;
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

.btn-small {
  padding: 4px 8px;
  font-size: 12px;
}
</style>

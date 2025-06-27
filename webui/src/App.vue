<script setup lang="ts">
import { ref, computed, onMounted } from 'vue'
import { useMainStore } from './stores/main'
import DashboardView from './components/DashboardView.vue'
import GenerateView from './components/GenerateView.vue'
import BindingsView from './components/BindingsView.vue'
import PersonalitiesView from './components/PersonalitiesView.vue'
import ModelsView from './components/ModelsView.vue'
import FunctionsView from './components/FunctionsView.vue'
import ConfigView from './components/ConfigView.vue'

const store = useMainStore()
const currentView = ref('dashboard')

// Define all possible views with their requirements
const allViews = [
  { id: 'dashboard', label: 'Dashboard', icon: '📊', requiresAuth: false },
  { id: 'generate', label: 'Generate', icon: '🚀', requiresAuth: true, feature: 'generate' },
  { id: 'bindings', label: 'Bindings', icon: '🔌', requiresAuth: true, feature: 'list_bindings' },
  { id: 'personalities', label: 'Personalities', icon: '👤', requiresAuth: true, feature: 'list_personalities' },
  { id: 'models', label: 'Models', icon: '🤖', requiresAuth: true, feature: 'list_models' },
  { id: 'functions', label: 'Functions', icon: '⚙️', requiresAuth: true, feature: 'list_functions' },
  { id: 'config', label: 'Config', icon: '🔧', requiresAuth: true }
]

// Only show views that are available based on connection and features
const availableViews = computed(() => {
  return allViews.filter(view => {
    // Dashboard is always available
    if (view.id === 'dashboard') return true

    // Check if authentication is required and available
    if (view.requiresAuth && !store.canMakeAuthenticatedRequests) return false

    // Check if specific feature is required and available
    if (view.feature && !store.availableFeatures.has(view.feature)) return false

    return true
  })
})

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
    case 'auth_required': return 'Auth Required'
    case 'error': return 'Error'
    default: return 'Disconnected'
  }
})

onMounted(() => {
  store.initialize()
})
</script>

<template>
  <div class="app">
    <nav class="nav">
      <div class="nav-brand">
        <span class="brand-text">LOLLMS Server</span>
        <span class="brand-status" :class="statusClass">{{ statusText }}</span>
      </div>
      <div class="nav-items">
        <button
          v-for="view in availableViews"
          :key="view.id"
          @click="currentView = view.id"
          :class="{ active: currentView === view.id }"
          class="nav-btn"
          :disabled="view.requiresAuth && !store.canMakeAuthenticatedRequests"
        >
          <span class="nav-icon">{{ view.icon }}</span>
          <span class="nav-label">{{ view.label }}</span>
        </button>
      </div>
      <div class="nav-info">
        <div class="connection-indicator" :class="statusClass"></div>
        <div class="feature-count" v-if="store.availableFeatures.size > 0">
          {{ store.availableFeatures.size }} features
        </div>
      </div>
    </nav>

    <main class="main">
      <DashboardView v-if="currentView === 'dashboard'" />
      <GenerateView v-else-if="currentView === 'generate'" />
      <BindingsView v-else-if="currentView === 'bindings'" />
      <PersonalitiesView v-else-if="currentView === 'personalities'" />
      <ModelsView v-else-if="currentView === 'models'" />
      <FunctionsView v-else-if="currentView === 'functions'" />
      <ConfigView v-else-if="currentView === 'config'" />
    </main>

    <!-- Global temporary message display -->
    <div v-if="store.temporaryMessage" :class="['global-message', `message-${store.temporaryMessage.type}`]">
      {{ store.temporaryMessage.text }}
    </div>
  </div>
</template>

<style>
* {
  margin: 0;
  padding: 0;
  box-sizing: border-box;
}

body {
  font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
  background: #1a237e;
  color: white;
  overflow-x: hidden;
}

.app {
  min-height: 100vh;
  background: #1a237e;
}

.nav {
  background: #0d47a1;
  display: flex;
  align-items: center;
  padding: 8px 16px;
  border-bottom: 1px solid #1565c0;
  min-height: 48px;
}

.nav-brand {
  display: flex;
  flex-direction: column;
  margin-right: 24px;
}

.brand-text {
  font-size: 16px;
  font-weight: bold;
  color: white;
}

.brand-status {
  font-size: 10px;
  font-weight: normal;
}

.nav-items {
  display: flex;
  flex: 1;
  gap: 2px;
}

.nav-btn {
  background: transparent;
  border: none;
  color: white;
  padding: 6px 12px;
  cursor: pointer;
  font-size: 12px;
  transition: background-color 0.2s;
  display: flex;
  align-items: center;
  gap: 4px;
}

.nav-btn:hover:not(:disabled) {
  background: #1565c0;
}

.nav-btn.active {
  background: #1976d2;
}

.nav-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.nav-icon {
  font-size: 14px;
}

.nav-label {
  font-size: 12px;
}

.nav-info {
  margin-left: auto;
  display: flex;
  align-items: center;
  gap: 8px;
}

.connection-indicator {
  width: 8px;
  height: 8px;
  border-radius: 50%;
}

.feature-count {
  font-size: 10px;
  color: #bbbbbb;
}

.status-connected { color: #4caf50; }
.status-connecting { color: #ff9800; }
.status-auth { color: #2196f3; }
.status-error { color: #f44336; }
.status-disconnected { color: #666666; }

.connection-indicator.status-connected { background: #4caf50; }
.connection-indicator.status-connecting { background: #ff9800; animation: pulse 1s infinite; }
.connection-indicator.status-auth { background: #2196f3; }
.connection-indicator.status-error { background: #f44336; }
.connection-indicator.status-disconnected { background: #666666; }

@keyframes pulse {
  0%, 100% { opacity: 1; }
  50% { opacity: 0.5; }
}

.main {
  background: #1a237e;
  min-height: calc(100vh - 48px);
}

.global-message {
  position: fixed;
  top: 60px;
  right: 20px;
  padding: 12px 20px;
  font-weight: bold;
  z-index: 1000;
  max-width: 400px;
  animation: slideIn 0.3s ease-out;
}

.message-success {
  background: rgba(76, 175, 80, 0.9);
  border: 1px solid #4caf50;
  color: white;
}

.message-error {
  background: rgba(244, 67, 54, 0.9);
  border: 1px solid #f44336;
  color: white;
}

.message-info {
  background: rgba(25, 118, 210, 0.9);
  border: 1px solid #1976d2;
  color: white;
}

@keyframes slideIn {
  from {
    transform: translateX(100%);
    opacity: 0;
  }
  to {
    transform: translateX(0);
    opacity: 1;
  }
}
</style>
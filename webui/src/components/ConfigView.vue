<template>
  <div class="config">
    <div class="config-layout">
      <div class="config-sections">
        <div class="card">
          <div class="card-header">Configuration Sections</div>
          <div class="card-content">
            <div class="section-list">
              <div 
                v-for="section in configSections" 
                :key="section.id"
                @click="selectSection(section)"
                :class="{ active: selectedSection?.id === section.id }"
                class="section-item"
              >
                <div class="section-name">{{ section.name }}</div>
                <div class="section-description">{{ section.description }}</div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="config-editor">
        <div class="card">
          <div class="card-header">
            {{ selectedSection ? selectedSection.name : 'Select Configuration Section' }}
            <div class="config-actions">
              <button @click="loadConfig" class="btn-action">↻</button>
              <button @click="saveConfig" class="btn-action" :disabled="!hasChanges">💾</button>
            </div>
          </div>
          <div class="card-content">
            <div v-if="!selectedSection" class="placeholder">
              Select a configuration section from the list to edit settings
            </div>
            <div v-else class="config-form">
              <!-- Server Settings -->
              <div v-if="selectedSection.id === 'server'" class="form-section">
                <div class="form-group">
                  <label>Host:</label>
                  <input v-model="config.host" class="input" placeholder="0.0.0.0">
                </div>
                <div class="form-group">
                  <label>Port:</label>
                  <input v-model.number="config.port" type="number" class="input" placeholder="9601">
                </div>
                <div class="form-group">
                  <label>API Key Required:</label>
                  <input v-model="config.api_key_required" type="checkbox" class="checkbox">
                </div>
                <div class="form-group">
                  <label>Enable CORS:</label>
                  <input v-model="config.enable_cors" type="checkbox" class="checkbox">
                </div>
                <div class="form-group">
                  <label>Allowed Origins:</label>
                  <textarea v-model="allowedOriginsText" class="textarea" rows="3" placeholder="http://localhost:3000"></textarea>
                </div>
              </div>

              <!-- Paths Settings -->
              <div v-if="selectedSection.id === 'paths'" class="form-section">
                <div class="form-group">
                  <label>Models Path:</label>
                  <input v-model="config.models_path" class="input" placeholder="./models">
                </div>
                <div class="form-group">
                  <label>Personalities Path:</label>
                  <input v-model="config.personalities_path" class="input" placeholder="./zoos/personalities">
                </div>
                <div class="form-group">
                  <label>Bindings Path:</label>
                  <input v-model="config.bindings_path" class="input" placeholder="./zoos/bindings">
                </div>
                <div class="form-group">
                  <label>Functions Path:</label>
                  <input v-model="config.functions_path" class="input" placeholder="./zoos/functions">
                </div>
                <div class="form-group">
                  <label>Config Path:</label>
                  <input v-model="config.config_path" class="input" placeholder="./lollms_configs">
                </div>
              </div>

              <!-- Security Settings -->
              <div v-if="selectedSection.id === 'security'" class="form-section">
                <div class="form-group">
                  <label>API Key:</label>
                  <input v-model="config.api_key" type="password" class="input" placeholder="Enter API key">
                </div>
                <div class="form-group">
                  <label>Encrypt Config:</label>
                  <input v-model="config.encrypt_config" type="checkbox" class="checkbox">
                </div>
                <div class="form-group">
                  <label>Max Request Size (MB):</label>
                  <input v-model.number="config.max_request_size" type="number" class="input" placeholder="100">
                </div>
                <div class="form-group">
                  <label>Rate Limit (requests/minute):</label>
                  <input v-model.number="config.rate_limit" type="number" class="input" placeholder="60">
                </div>
              </div>

              <!-- Default Bindings -->
              <div v-if="selectedSection.id === 'defaults'" class="form-section">
                <div class="form-group">
                  <label>Default Text-to-Text Binding:</label>
                  <select v-model="config.default_ttt_binding" class="select">
                    <option value="">None</option>
                    <option v-for="binding in store.activeBindings" :key="binding.binding_instance_name" :value="binding.binding_instance_name">
                      {{ binding.binding_instance_name }}
                    </option>
                  </select>
                </div>
                <div class="form-group">
                  <label>Default Text-to-Image Binding:</label>
                  <select v-model="config.default_tti_binding" class="select">
                    <option value="">None</option>
                    <option v-for="binding in store.activeBindings" :key="binding.binding_instance_name" :value="binding.binding_instance_name">
                      {{ binding.binding_instance_name }}
                    </option>
                  </select>
                </div>
                <div class="form-group">
                  <label>Default Text-to-Speech Binding:</label>
                  <select v-model="config.default_tts_binding" class="select">
                    <option value="">None</option>
                    <option v-for="binding in store.activeBindings" :key="binding.binding_instance_name" :value="binding.binding_instance_name">
                      {{ binding.binding_instance_name }}
                    </option>
                  </select>
                </div>
                <div class="form-group">
                  <label>Default Speech-to-Text Binding:</label>
                  <select v-model="config.default_stt_binding" class="select">
                    <option value="">None</option>
                    <option v-for="binding in store.activeBindings" :key="binding.binding_instance_name" :value="binding.binding_instance_name">
                      {{ binding.binding_instance_name }}
                    </option>
                  </select>
                </div>
                <div class="form-group">
                  <label>Default Personality:</label>
                  <select v-model="config.default_personality" class="select">
                    <option value="">None</option>
                    <option v-for="personality in store.personalities" :key="personality.name" :value="personality.name">
                      {{ personality.name }}
                    </option>
                  </select>
                </div>
              </div>

              <!-- Resource Management -->
              <div v-if="selectedSection.id === 'resources'" class="form-section">
                <div class="form-group">
                  <label>Max Concurrent Requests:</label>
                  <input v-model.number="config.max_concurrent_requests" type="number" class="input" placeholder="10">
                </div>
                <div class="form-group">
                  <label>Request Timeout (seconds):</label>
                  <input v-model.number="config.request_timeout" type="number" class="input" placeholder="300">
                </div>
                <div class="form-group">
                  <label>GPU Memory Limit (GB):</label>
                  <input v-model.number="config.gpu_memory_limit" type="number" class="input" placeholder="8">
                </div>
                <div class="form-group">
                  <label>Enable GPU Monitoring:</label>
                  <input v-model="config.enable_gpu_monitoring" type="checkbox" class="checkbox">
                </div>
              </div>

              <!-- Logging Settings -->
              <div v-if="selectedSection.id === 'logging'" class="form-section">
                <div class="form-group">
                  <label>Log Level:</label>
                  <select v-model="config.log_level" class="select">
                    <option value="DEBUG">Debug</option>
                    <option value="INFO">Info</option>
                    <option value="WARNING">Warning</option>
                    <option value="ERROR">Error</option>
                    <option value="CRITICAL">Critical</option>
                  </select>
                </div>
                <div class="form-group">
                  <label>Log File Path:</label>
                  <input v-model="config.log_file_path" class="input" placeholder="./logs/lollms.log">
                </div>
                <div class="form-group">
                  <label>Max Log File Size (MB):</label>
                  <input v-model.number="config.max_log_file_size" type="number" class="input" placeholder="10">
                </div>
                <div class="form-group">
                  <label>Log Rotation Count:</label>
                  <input v-model.number="config.log_rotation_count" type="number" class="input" placeholder="5">
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>

    <div class="config-status">
      <div v-if="saveStatus" :class="saveStatus.type" class="status-message">
        {{ saveStatus.message }}
      </div>
    </div>
  </div>
</template>

<script setup lang="ts">
import { ref, reactive, computed, onMounted, watch } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()

const configSections = [
  { id: 'server', name: 'Server Settings', description: 'Host, port, and basic server configuration' },
  { id: 'paths', name: 'File Paths', description: 'Paths to models, personalities, bindings, and configs' },
  { id: 'security', name: 'Security', description: 'API keys, encryption, and access control' },
  { id: 'defaults', name: 'Default Bindings', description: 'Default bindings for different generation types' },
  { id: 'resources', name: 'Resource Management', description: 'Memory limits, concurrency, and timeouts' },
  { id: 'logging', name: 'Logging', description: 'Log levels, file paths, and rotation settings' }
]

const selectedSection = ref<any>(null)
const originalConfig = ref<any>({})
const saveStatus = ref<any>(null)

const config = reactive({
  // Server settings
  host: '0.0.0.0',
  port: 9601,
  api_key_required: true,
  enable_cors: true,
  allowed_origins: [] as string[],
  
  // Paths
  models_path: './models',
  personalities_path: './zoos/personalities',
  bindings_path: './zoos/bindings',
  functions_path: './zoos/functions',
  config_path: './lollms_configs',
  
  // Security
  api_key: '',
  encrypt_config: false,
  max_request_size: 100,
  rate_limit: 60,
  
  // Defaults
  default_ttt_binding: '',
  default_tti_binding: '',
  default_tts_binding: '',
  default_stt_binding: '',
  default_personality: '',
  
  // Resources
  max_concurrent_requests: 10,
  request_timeout: 300,
  gpu_memory_limit: 8,
  enable_gpu_monitoring: false,
  
  // Logging
  log_level: 'INFO',
  log_file_path: './logs/lollms.log',
  max_log_file_size: 10,
  log_rotation_count: 5
})

const allowedOriginsText = computed({
  get: () => config.allowed_origins.join('\n'),
  set: (value: string) => {
    config.allowed_origins = value.split('\n').filter(origin => origin.trim() !== '')
  }
})

const hasChanges = computed(() => {
  return JSON.stringify(config) !== JSON.stringify(originalConfig.value)
})

onMounted(() => {
  selectSection(configSections[0])
  loadConfig()
  if (store.activeBindings.length === 0) {
    store.loadBindings()
  }
  if (store.personalities.length === 0) {
    store.loadPersonalities()
  }
})

function selectSection(section: any) {
  selectedSection.value = section
}

function loadConfig() {
  // In a real implementation, this would load from the server
  // For now, we'll use default values
  originalConfig.value = JSON.parse(JSON.stringify(config))
  saveStatus.value = null
}

function saveConfig() {
  // Simulate saving configuration
  saveStatus.value = { type: 'success', message: 'Configuration saved successfully!' }
  originalConfig.value = JSON.parse(JSON.stringify(config))
  
  setTimeout(() => {
    saveStatus.value = null
  }, 3000)
}

// Clear status when config changes
watch(config, () => {
  if (saveStatus.value?.type === 'success') {
    saveStatus.value = null
  }
}, { deep: true })
</script>

<style scoped>
.config {
  background: #1a237e;
  color: white;
}

.config-layout {
  display: grid;
  grid-template-columns: 300px 1fr;
  gap: 16px;
  height: calc(100vh - 120px);
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

.config-actions {
  display: flex;
  gap: 8px;
}

.btn-action {
  background: transparent;
  border: none;
  color: white;
  cursor: pointer;
  font-size: 16px;
  padding: 4px;
}

.btn-action:disabled {
  color: #666666;
  cursor: not-allowed;
}

.card-content {
  padding: 16px;
  flex: 1;
  overflow-y: auto;
}

.section-item {
  padding: 12px;
  border: 1px solid #1565c0;
  margin-bottom: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.section-item:hover {
  background: #1565c0;
}

.section-item.active {
  background: #1976d2;
  border-color: #2196f3;
}

.section-name {
  font-weight: bold;
  margin-bottom: 4px;
}

.section-description {
  font-size: 12px;
  color: #cccccc;
}

.placeholder {
  text-align: center;
  color: #bbbbbb;
  padding: 32px;
}

.form-section {
  display: flex;
  flex-direction: column;
  gap: 16px;
}

.form-group {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.form-group label {
  font-size: 14px;
  font-weight: bold;
  color: #bbbbbb;
}

.input, .select, .textarea {
  background: #1a237e;
  border: 1px solid #1565c0;
  color: white;
  padding: 8px 12px;
  font-size: 14px;
}

.input:focus, .select:focus, .textarea:focus {
  outline: none;
  border-color: #1976d2;
}

.checkbox {
  width: auto;
}

.textarea {
  resize: vertical;
  min-height: 60px;
}

.config-status {
  padding: 16px;
}

.status-message {
  padding: 12px 16px;
  font-weight: bold;
  text-align: center;
}

.status-message.success {
  background: rgba(76, 175, 80, 0.2);
  border: 1px solid #4caf50;
  color: #4caf50;
}

.status-message.error {
  background: rgba(244, 67, 54, 0.2);
  border: 1px solid #f44336;
  color: #f44336;
}
</style>

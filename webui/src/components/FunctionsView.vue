<template>
  <div class="functions">
    <div class="functions-layout">
      <div class="functions-list">
        <div class="card">
          <div class="card-header">
            Available Functions
            <button @click="store.loadFunctions()" class="btn-refresh">↻</button>
          </div>
          <div class="card-content">
            <div v-if="store.loadingStates.functions" class="loading">Loading functions...</div>
            <div v-else-if="store.functions.length === 0" class="empty">
              No custom functions found
            </div>
            <div v-else class="function-list">
              <div 
                v-for="func in store.functions" 
                :key="func.name"
                @click="selectFunction(func)"
                :class="{ active: selectedFunction?.name === func.name }"
                class="function-item"
              >
                <div class="function-header">
                  <div class="function-name">{{ func.name }}</div>
                  <div class="function-module">{{ func.module }}</div>
                </div>
                <div class="function-signature">{{ func.signature }}</div>
                <div class="function-description">
                  {{ func.description || 'No description available' }}
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="function-details">
        <div class="card">
          <div class="card-header">
            {{ selectedFunction ? 'Function Details' : 'Select a Function' }}
          </div>
          <div class="card-content">
            <div v-if="!selectedFunction" class="placeholder">
              Select a function from the list to view details
            </div>
            <div v-else class="details">
              <div class="detail-section">
                <h3>Basic Information</h3>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="label">Name:</span>
                    <span class="value">{{ selectedFunction.name }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Module:</span>
                    <span class="value">{{ selectedFunction.module }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Full Name:</span>
                    <span class="value">{{ selectedFunction.module }}.{{ selectedFunction.name }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Type:</span>
                    <span class="value">{{ selectedFunction.is_async ? 'Async Function' : 'Sync Function' }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Function Signature</h3>
                <div class="signature-display">
                  <code>{{ selectedFunction.signature }}</code>
                </div>
              </div>

              <div class="detail-section" v-if="selectedFunction.description">
                <h3>Description</h3>
                <div class="description-text">
                  {{ selectedFunction.description }}
                </div>
              </div>

              <div class="detail-section" v-if="selectedFunction.parameters && selectedFunction.parameters.length > 0">
                <h3>Parameters</h3>
                <div class="parameters-list">
                  <div v-for="param in selectedFunction.parameters" :key="param.name" class="parameter-item">
                    <div class="parameter-header">
                      <span class="parameter-name">{{ param.name }}</span>
                      <span class="parameter-type">{{ param.type || 'Any' }}</span>
                      <span v-if="param.required" class="parameter-required">Required</span>
                      <span v-else class="parameter-optional">Optional</span>
                    </div>
                    <div v-if="param.description" class="parameter-description">
                      {{ param.description }}
                    </div>
                    <div v-if="param.default !== undefined" class="parameter-default">
                      Default: <code>{{ param.default }}</code>
                    </div>
                  </div>
                </div>
              </div>

              <div class="detail-section" v-if="selectedFunction.return_type">
                <h3>Return Type</h3>
                <div class="return-type">
                  <code>{{ selectedFunction.return_type }}</code>
                </div>
              </div>

              <div class="detail-section">
                <h3>Usage Example</h3>
                <div class="usage-example">
                  <code>{{ generateUsageExample(selectedFunction) }}</code>
                </div>
              </div>

              <div class="detail-section">
                <h3>Function Testing</h3>
                <div class="test-section">
                  <div class="test-form">
                    <div v-if="selectedFunction.parameters && selectedFunction.parameters.length > 0">
                      <div v-for="param in selectedFunction.parameters" :key="param.name" class="test-param">
                        <label>{{ param.name }} ({{ param.type || 'Any' }}):</label>
                        <input 
                          v-model="testParams[param.name]" 
                          :placeholder="param.description || `Enter ${param.name}`"
                          class="test-input"
                        />
                      </div>
                    </div>
                    <div class="test-actions">
                      <button @click="testFunction" class="btn btn-primary" :disabled="testing">
                        {{ testing ? 'Testing...' : 'Test Function' }}
                      </button>
                      <button @click="clearTestParams" class="btn btn-secondary">
                        Clear
                      </button>
                    </div>
                  </div>
                  <div v-if="testResult" class="test-result">
                    <h4>Test Result:</h4>
                    <pre>{{ testResult }}</pre>
                  </div>
                  <div v-if="testError" class="test-error">
                    <h4>Test Error:</h4>
                    <pre>{{ testError }}</pre>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Integration</h3>
                <div class="integration-info">
                  <div class="integration-item">
                    <strong>Personality Usage:</strong>
                    <p>This function can be called from scripted personalities using the function manager.</p>
                  </div>
                  <div class="integration-item">
                    <strong>Call Format:</strong>
                    <code>await context['function_manager'].execute_function('{{ selectedFunction.module }}.{{ selectedFunction.name }}', parameters)</code>
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
import { ref, reactive, onMounted } from 'vue'
import { useMainStore } from '../stores/main'

const store = useMainStore()
const selectedFunction = ref<any>(null)
const testParams = reactive<Record<string, any>>({})
const testResult = ref('')
const testError = ref('')
const testing = ref(false)

onMounted(() => {
  if (store.functions.length === 0) {
    store.loadFunctions()
  }
})

function selectFunction(func: any) {
  selectedFunction.value = func
  clearTestParams()
  testResult.value = ''
  testError.value = ''
}

function clearTestParams() {
  Object.keys(testParams).forEach(key => {
    delete testParams[key]
  })
}

function generateUsageExample(func: any): string {
  const params = func.parameters || []
  const paramList = params.map((p: any) => `${p.name}=${p.default !== undefined ? p.default : `"${p.name}_value"`}`).join(', ')
  return `await context['function_manager'].execute_function('${func.module}.${func.name}', {${paramList}})`
}

async function testFunction() {
  if (!selectedFunction.value) return
  
  testing.value = true
  testResult.value = ''
  testError.value = ''
  
  try {
    // This would need to be implemented in the store
    // For now, we'll simulate a test
    const result = await simulateFunction(selectedFunction.value, testParams)
    testResult.value = JSON.stringify(result, null, 2)
  } catch (error: any) {
    testError.value = error.message || 'Function test failed'
  } finally {
    testing.value = false
  }
}

async function simulateFunction(func: any, params: Record<string, any>): Promise<any> {
  // Simulate function execution
  await new Promise(resolve => setTimeout(resolve, 1000))
  
  // Mock response based on function name
  if (func.name.includes('calculate')) {
    return { result: 42, calculation: 'simulated' }
  } else if (func.name.includes('fetch')) {
    return { data: 'simulated data', status: 'success' }
  } else {
    return { message: 'Function executed successfully', params }
  }
}
</script>

<style scoped>
.functions {
  background: #1a237e;
  color: white;
}

.functions-layout {
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

.function-item {
  padding: 12px;
  border: 1px solid #1565c0;
  margin-bottom: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.function-item:hover {
  background: #1565c0;
}

.function-item.active {
  background: #1976d2;
  border-color: #2196f3;
}

.function-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.function-name {
  font-weight: bold;
}

.function-module {
  font-size: 12px;
  color: #bbbbbb;
}

.function-signature {
  font-family: monospace;
  font-size: 12px;
  color: #4caf50;
  margin-bottom: 4px;
}

.function-description {
  font-size: 12px;
  color: #cccccc;
  line-height: 1.4;
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

.signature-display, .return-type, .usage-example {
  background: rgba(76, 175, 80, 0.1);
  border: 1px solid #4caf50;
  padding: 12px;
  font-family: monospace;
  font-size: 14px;
}

.description-text {
  background: rgba(25, 118, 210, 0.1);
  border: 1px solid #1976d2;
  padding: 12px;
  line-height: 1.5;
}

.parameters-list {
  display: flex;
  flex-direction: column;
  gap: 12px;
}

.parameter-item {
  border: 1px solid #1565c0;
  padding: 12px;
}

.parameter-header {
  display: flex;
  gap: 8px;
  align-items: center;
  margin-bottom: 4px;
}

.parameter-name {
  font-weight: bold;
}

.parameter-type {
  background: #1976d2;
  padding: 2px 6px;
  font-size: 10px;
}

.parameter-required {
  background: #f44336;
  padding: 2px 6px;
  font-size: 10px;
}

.parameter-optional {
  background: #4caf50;
  padding: 2px 6px;
  font-size: 10px;
}

.parameter-description {
  font-size: 12px;
  color: #cccccc;
  margin-bottom: 4px;
}

.parameter-default {
  font-size: 12px;
  color: #bbbbbb;
}

.test-section {
  background: rgba(66, 66, 66, 0.3);
  border: 1px solid #666666;
  padding: 16px;
}

.test-param {
  margin-bottom: 12px;
}

.test-param label {
  display: block;
  margin-bottom: 4px;
  font-size: 12px;
  font-weight: bold;
}

.test-input {
  width: 100%;
  background: #1a237e;
  border: 1px solid #1565c0;
  color: white;
  padding: 8px 12px;
  font-size: 14px;
}

.test-actions {
  display: flex;
  gap: 8px;
  margin-bottom: 16px;
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

.test-result {
  background: rgba(76, 175, 80, 0.1);
  border: 1px solid #4caf50;
  padding: 12px;
}

.test-error {
  background: rgba(244, 67, 54, 0.1);
  border: 1px solid #f44336;
  padding: 12px;
}

.test-result h4, .test-error h4 {
  margin-bottom: 8px;
  color: inherit;
}

.test-result pre, .test-error pre {
  font-family: monospace;
  font-size: 12px;
  white-space: pre-wrap;
  margin: 0;
}

.integration-info {
  background: rgba(25, 118, 210, 0.1);
  border: 1px solid #1976d2;
  padding: 16px;
}

.integration-item {
  margin-bottom: 12px;
}

.integration-item:last-child {
  margin-bottom: 0;
}

.integration-item strong {
  color: #2196f3;
}

.integration-item p {
  margin: 4px 0;
  font-size: 14px;
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

import { ref, computed } from 'vue'
import { defineStore } from 'pinia'
import axios, { AxiosError } from 'axios'

// Connection states
type ConnectionState = 'disconnected' | 'connecting' | 'connected' | 'error' | 'auth_required'

// API response types based on actual LOLLMS Server models
interface HealthResponse {
  status: string
  version: string
  api_key_required: boolean
}

interface ApiError {
  detail: string
  status_code: number
  timestamp: string
}

export const useMainStore = defineStore('main', () => {
  // Connection state management
  const connectionState = ref<ConnectionState>('disconnected')
  const serverInfo = ref<HealthResponse | null>(null)
  const apiKey = ref(localStorage.getItem('lollms_api_key') || '')
  const lastError = ref<string>('')

  // Feature availability (discovered progressively)
  const availableFeatures = ref<Set<string>>(new Set())

  // Data from working endpoints
  const bindingTypes = ref<any[]>([])
  const activeBindings = ref<any[]>([])
  const personalities = ref<any[]>([])
  const functions = ref<any[]>([])
  const discoveredModels = ref<Record<string, string[]>>({})
  const defaultBindings = ref<any>({})

  // Loading states for different operations
  const loadingStates = ref<Record<string, boolean>>({})

  // Computed properties
  const isConnected = computed(() => connectionState.value === 'connected')
  const needsApiKey = computed(() => serverInfo.value?.api_key_required && !apiKey.value)
  const canMakeAuthenticatedRequests = computed(() =>
    isConnected.value && (!serverInfo.value?.api_key_required || apiKey.value)
  )

  // Create axios instance
  const createApiClient = () => {
    const headers: Record<string, string> = {
      'Content-Type': 'application/json'
    }

    if (apiKey.value) {
      headers['X-API-Key'] = apiKey.value
    }

    return axios.create({
      baseURL: '/api/v1',
      headers,
      timeout: 10000
    })
  }

  // Error handling utility
  function handleApiError(error: unknown, operation: string): string {
    console.error(`${operation} failed:`, error)

    if (axios.isAxiosError(error)) {
      const axiosError = error as AxiosError<ApiError>

      if (axiosError.response?.status === 401 || axiosError.response?.status === 403) {
        connectionState.value = 'auth_required'
        return 'Authentication required. Please check your API key.'
      }

      if (axiosError.response?.data?.detail) {
        return axiosError.response.data.detail
      }

      if (axiosError.message) {
        return axiosError.message
      }
    }

    return `${operation} failed: ${error}`
  }

  // Set loading state for specific operation
  function setLoading(operation: string, loading: boolean) {
    loadingStates.value[operation] = loading
  }

  // Update API key and recreate client
  function setApiKey(key: string) {
    apiKey.value = key.trim()
    localStorage.setItem('lollms_api_key', apiKey.value)

    // Clear auth error state if we now have a key
    if (apiKey.value && connectionState.value === 'auth_required') {
      connectionState.value = 'connected'
    }
  }

  // Health check - the foundation of everything
  async function checkHealth(): Promise<boolean> {
    setLoading('health', true)
    connectionState.value = 'connecting'
    lastError.value = ''

    try {
      const response = await axios.get('/health', { timeout: 5000 })
      serverInfo.value = response.data as HealthResponse
      connectionState.value = 'connected'

      // Mark health as available feature
      availableFeatures.value.add('health')

      return true
    } catch (error) {
      connectionState.value = 'error'
      lastError.value = handleApiError(error, 'Health check')
      serverInfo.value = null
      return false
    } finally {
      setLoading('health', false)
    }
  }

  // Test if an endpoint is available
  async function testEndpoint(endpoint: string): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    try {
      const api = createApiClient()
      await api.get(endpoint)
      availableFeatures.value.add(endpoint)
      return true
    } catch (error) {
      console.warn(`Endpoint ${endpoint} not available:`, error)
      return false
    }
  }

  // Load bindings (both types and active instances)
  async function loadBindings(): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    setLoading('bindings', true)

    try {
      const api = createApiClient()

      // Try to get binding types
      try {
        const typesResponse = await api.get('/list_bindings')
        bindingTypes.value = typesResponse.data.binding_types || []
        availableFeatures.value.add('list_bindings')
      } catch (error) {
        console.warn('Could not load binding types:', error)
        bindingTypes.value = []
      }

      // Try to get active bindings
      try {
        const activeResponse = await api.get('/list_active_bindings')
        activeBindings.value = activeResponse.data.binding_instances || []
        availableFeatures.value.add('list_active_bindings')
      } catch (error) {
        console.warn('Could not load active bindings:', error)
        activeBindings.value = []
      }

      return true
    } catch (error) {
      lastError.value = handleApiError(error, 'Load bindings')
      return false
    } finally {
      setLoading('bindings', false)
    }
  }

  // Load personalities
  async function loadPersonalities(): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    setLoading('personalities', true)

    try {
      const api = createApiClient()
      const response = await api.get('/list_personalities')
      personalities.value = Object.values(response.data.personalities || {})
      availableFeatures.value.add('list_personalities')
      return true
    } catch (error) {
      lastError.value = handleApiError(error, 'Load personalities')
      personalities.value = []
      return false
    } finally {
      setLoading('personalities', false)
    }
  }

  // Load functions
  async function loadFunctions(): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    setLoading('functions', true)

    try {
      const api = createApiClient()
      const response = await api.get('/list_functions')
      functions.value = response.data.functions || []
      availableFeatures.value.add('list_functions')
      return true
    } catch (error) {
      lastError.value = handleApiError(error, 'Load functions')
      functions.value = []
      return false
    } finally {
      setLoading('functions', false)
    }
  }

  // Load discovered models (file scan)
  async function loadDiscoveredModels(): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    setLoading('models', true)

    try {
      const api = createApiClient()
      const response = await api.get('/list_models')
      discoveredModels.value = response.data.models || {}
      availableFeatures.value.add('list_models')
      return true
    } catch (error) {
      lastError.value = handleApiError(error, 'Load discovered models')
      discoveredModels.value = {}
      return false
    } finally {
      setLoading('models', false)
    }
  }

  // Load default bindings configuration
  async function loadDefaultBindings(): Promise<boolean> {
    if (!canMakeAuthenticatedRequests.value) return false

    setLoading('defaults', true)

    try {
      const api = createApiClient()
      const response = await api.get('/get_default_bindings')
      defaultBindings.value = response.data
      availableFeatures.value.add('get_default_bindings')
      return true
    } catch (error) {
      lastError.value = handleApiError(error, 'Load default bindings')
      defaultBindings.value = {}
      return false
    } finally {
      setLoading('defaults', false)
    }
  }

  // Generate content
  async function generate(request: any): Promise<any> {
    if (!canMakeAuthenticatedRequests.value) {
      throw new Error('Authentication required for generation')
    }

    setLoading('generate', true)

    try {
      const api = createApiClient()
      const response = await api.post('/generate', request)
      availableFeatures.value.add('generate')
      return response.data
    } catch (error) {
      const errorMsg = handleApiError(error, 'Generate content')
      throw new Error(errorMsg)
    } finally {
      setLoading('generate', false)
    }
  }

  // Get model info for specific binding
  async function getModelInfo(bindingInstanceName: string): Promise<any> {
    if (!canMakeAuthenticatedRequests.value) {
      throw new Error('Authentication required')
    }

    try {
      const api = createApiClient()
      const response = await api.get(`/get_model_info/${bindingInstanceName}`)
      return response.data
    } catch (error) {
      const errorMsg = handleApiError(error, 'Get model info')
      throw new Error(errorMsg)
    }
  }

  // Get available models for specific binding
  async function getAvailableModels(bindingInstanceName: string): Promise<any> {
    if (!canMakeAuthenticatedRequests.value) {
      throw new Error('Authentication required')
    }

    try {
      const api = createApiClient()
      const response = await api.get(`/list_available_models/${bindingInstanceName}`)
      return response.data
    } catch (error) {
      const errorMsg = handleApiError(error, 'Get available models')
      throw new Error(errorMsg)
    }
  }

  // Initialize connection and discover features
  async function initialize(): Promise<void> {
    const healthOk = await checkHealth()

    if (healthOk && canMakeAuthenticatedRequests.value) {
      // Try to load all available data
      await Promise.allSettled([
        loadBindings(),
        loadPersonalities(),
        loadFunctions(),
        loadDiscoveredModels(),
        loadDefaultBindings()
      ])
    }
  }

  // Clear all data (for disconnection/logout)
  function clearData() {
    bindingTypes.value = []
    activeBindings.value = []
    personalities.value = []
    functions.value = []
    discoveredModels.value = {}
    defaultBindings.value = {}
    availableFeatures.value.clear()
    lastError.value = ''
  }

  return {
    // State
    connectionState,
    serverInfo,
    apiKey,
    lastError,
    availableFeatures,
    bindingTypes,
    activeBindings,
    personalities,
    functions,
    discoveredModels,
    defaultBindings,
    loadingStates,

    // Computed
    isConnected,
    needsApiKey,
    canMakeAuthenticatedRequests,

    // Actions
    setApiKey,
    checkHealth,
    initialize,
    clearData,
    testEndpoint,
    loadBindings,
    loadPersonalities,
    loadFunctions,
    loadDiscoveredModels,
    loadDefaultBindings,
    generate,
    getModelInfo,
    getAvailableModels
  }
})
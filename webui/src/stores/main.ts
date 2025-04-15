// webui/src/stores/main.ts
import { defineStore } from 'pinia'
import { ref, computed } from 'vue'
// Import your JS client library OR use fetch/axios directly
// Assuming you might place a refined client lib here:
// import { LollmsClient } from '@/services/lollms-client' // Example path

// If using axios: import axios from 'axios';
// const API_BASE = '/api/v1'; // Relative path for API calls

export const useMainStore = defineStore('main', () => {
  // State refs
  const serverConnected = ref(false)
  const personalities = ref<any[]>([]) // Replace 'any' with proper type later
  const bindings = ref<any[]>([])      // Replace 'any' with proper type later
  const loadingError = ref<string | null>(null)

  // Getters (computed refs)
  const personalityNames = computed(() => personalities.value.map(p => p.name))
  const bindingNames = computed(() => bindings.value.map(b => b.name)) // Assuming bindings have names

  // Actions (async functions)
  async function initialize(baseUrl: string, apiKey?: string) {
    loadingError.value = null;
    serverConnected.value = false;
    // --- Use fetch or axios, or instantiate your JS client ---
    // Example using fetch:
    const headers: HeadersInit = {"Content-Type": "application/json"};
    if (apiKey) headers['X-API-Key'] = apiKey;

    try {
        const [bindingsRes, personalitiesRes] = await Promise.all([
             fetch(`${baseUrl}/api/v1/list_bindings`, { method: 'GET', headers }),
             fetch(`${baseUrl}/api/v1/list_personalities`, { method: 'GET', headers })
        ]);

        if (!bindingsRes.ok) throw new Error(`Bindings fetch failed: ${bindingsRes.statusText}`);
        if (!personalitiesRes.ok) throw new Error(`Personalities fetch failed: ${personalitiesRes.statusText}`);

        const bindingsData = await bindingsRes.json();
        const personalitiesData = await personalitiesRes.json();

        // Extract relevant data (adjust based on actual API response structure)
        bindings.value = bindingsData.binding_instances ? Object.values(bindingsData.binding_instances) : [];
        personalities.value = personalitiesData.personalities ? Object.values(personalitiesData.personalities) : [];

        serverConnected.value = true;
        console.log("Loaded bindings:", bindings.value);
        console.log("Loaded personalities:", personalities.value);

    } catch (error: any) {
        console.error("Initialization failed:", error);
        loadingError.value = error.message || "Failed to connect or load data.";
        serverConnected.value = false;
    }
    // --- End API call example ---
  }

  return {
    // State
    serverConnected,
    personalities,
    bindings,
    loadingError,
    // Getters
    personalityNames,
    bindingNames,
    // Actions
    initialize,
  }
})
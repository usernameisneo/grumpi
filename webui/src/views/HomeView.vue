<script setup lang="ts">
import { ref } from 'vue'
import { useMainStore } from '@/stores/main'

const store = useMainStore()
const serverUrl = ref('http://localhost:9600') // Default URL
const apiKey = ref('')

async function connect() {
  await store.initialize(serverUrl.value, apiKey.value || undefined);
}
</script>

<template>
  <main>
    <h2>Connect to Server</h2>
    <div class="control-group">
       <label for="home-server-url">Server URL:</label>
       <input type="text" id="home-server-url" v-model="serverUrl">
    </div>
     <div class="control-group">
       <label for="home-api-key">API Key (Optional):</label>
       <input type="password" id="home-api-key" v-model="apiKey" placeholder="Enter API Key if required">
    </div>
    <button @click="connect" :disabled="store.serverConnected">
      {{ store.serverConnected ? 'Connected' : 'Connect & Load' }}
    </button>
    <p v-if="store.loadingError" class="status error">Error: {{ store.loadingError }}</p>
    <p v-if="store.serverConnected" class="status success">
        Successfully connected! Found {{ store.bindings.length }} bindings and {{ store.personalities.length }} personalities.
        Go to <RouterLink to="/chat">Chat</RouterLink>.
    </p>
  </main>
</template>

<style scoped>
  /* Add specific styles if needed */
  .control-group { margin-bottom: 1rem; }
  label { display: block; margin-bottom: 0.25rem;}
  input { width: 300px; padding: 5px; margin-bottom: 0.5rem;}
  button:disabled { opacity: 0.6; cursor: default;}
   .status { margin-top: 1rem; font-style: italic; }
   .error { color: red; }
   .success { color: green; }
</style>
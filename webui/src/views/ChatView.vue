<script setup lang="ts">
import { ref, onMounted } from 'vue'
import { useMainStore } from '@/stores/main' // Import Pinia store

const store = useMainStore()
const message = ref('Welcome to the Chat View!')

onMounted(() => {
  // Example: Access store data when component mounts
  if (!store.serverConnected) {
    message.value = 'Server not connected. Please connect on Home page first.'
    // Optionally try to initialize if not connected
    // store.initialize('http://localhost:9600'); // Example: Adjust URL/API key logic
  } else {
     message.value = `Connected! Found ${store.personalities.length} personalities.`
  }
});

// Add chat input, message display, generation logic here later
</script>

<template>
  <div class="chat-view">
    <h2>Chat Interface</h2>
    <p>{{ message }}</p>
    <!-- Placeholder for chat elements -->
    <div>
        <textarea placeholder="Enter your message..."></textarea>
        <button>Send</button>
    </div>
    <div class="message-list">
        <!-- Chat messages will appear here -->
    </div>
  </div>
</template>

<style scoped>
  .chat-view {
    padding: 1rem;
  }
  textarea { width: 90%; min-height: 50px; margin-bottom: 5px;}
  .message-list { margin-top: 1rem; border: 1px solid #eee; min-height: 200px; padding: 10px;}
</style>
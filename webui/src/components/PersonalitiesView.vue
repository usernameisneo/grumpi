<template>
  <div class="personalities">
    <div class="personalities-layout">
      <div class="personalities-list">
        <div class="card">
          <div class="card-header">
            Available Personalities
            <button @click="store.loadPersonalities()" class="btn-refresh">↻</button>
          </div>
          <div class="card-content">
            <div v-if="store.loadingStates.personalities" class="loading">Loading personalities...</div>
            <div v-else-if="store.personalities.length === 0" class="empty">
              No personalities found
            </div>
            <div v-else class="personality-list">
              <div 
                v-for="personality in store.personalities" 
                :key="personality.name"
                @click="selectPersonality(personality)"
                :class="{ active: selectedPersonality?.name === personality.name }"
                class="personality-item"
              >
                <div class="personality-header">
                  <div class="personality-name">{{ personality.name }}</div>
                  <div class="personality-version">v{{ personality.version }}</div>
                </div>
                <div class="personality-author">{{ personality.author }}</div>
                <div class="personality-category">{{ personality.category || 'General' }}</div>
                <div class="personality-description">
                  {{ personality.personality_description?.substring(0, 100) }}...
                </div>
                <div class="personality-tags">
                  <span v-for="tag in personality.tags?.slice(0, 3)" :key="tag" class="tag">
                    {{ tag }}
                  </span>
                </div>
              </div>
            </div>
          </div>
        </div>
      </div>

      <div class="personality-details">
        <div class="card">
          <div class="card-header">
            {{ selectedPersonality ? 'Personality Details' : 'Select a Personality' }}
          </div>
          <div class="card-content">
            <div v-if="!selectedPersonality" class="placeholder">
              Select a personality from the list to view details
            </div>
            <div v-else class="details">
              <div class="detail-section">
                <h3>Basic Information</h3>
                <div class="detail-grid">
                  <div class="detail-item">
                    <span class="label">Name:</span>
                    <span class="value">{{ selectedPersonality.name }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Author:</span>
                    <span class="value">{{ selectedPersonality.author }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Version:</span>
                    <span class="value">{{ selectedPersonality.version }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Category:</span>
                    <span class="value">{{ selectedPersonality.category || 'General' }}</span>
                  </div>
                  <div class="detail-item">
                    <span class="label">Language:</span>
                    <span class="value">{{ selectedPersonality.language || 'English' }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedPersonality.script_path">
                    <span class="label">Script:</span>
                    <span class="value">{{ selectedPersonality.script_path }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section">
                <h3>Description</h3>
                <div class="description-text">
                  {{ selectedPersonality.personality_description }}
                </div>
              </div>

              <div class="detail-section">
                <h3>System Conditioning</h3>
                <div class="conditioning-text">
                  {{ selectedPersonality.personality_conditioning }}
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.welcome_message">
                <h3>Welcome Message</h3>
                <div class="welcome-text">
                  {{ selectedPersonality.welcome_message }}
                </div>
              </div>

              <div class="detail-section">
                <h3>Model Parameters</h3>
                <div class="detail-grid">
                  <div class="detail-item" v-if="selectedPersonality.model_temperature !== null">
                    <span class="label">Temperature:</span>
                    <span class="value">{{ selectedPersonality.model_temperature }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedPersonality.model_n_predicts">
                    <span class="label">Max Tokens:</span>
                    <span class="value">{{ selectedPersonality.model_n_predicts }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedPersonality.model_top_k">
                    <span class="label">Top K:</span>
                    <span class="value">{{ selectedPersonality.model_top_k }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedPersonality.model_top_p">
                    <span class="label">Top P:</span>
                    <span class="value">{{ selectedPersonality.model_top_p }}</span>
                  </div>
                  <div class="detail-item" v-if="selectedPersonality.model_repeat_penalty">
                    <span class="label">Repeat Penalty:</span>
                    <span class="value">{{ selectedPersonality.model_repeat_penalty }}</span>
                  </div>
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.tags?.length">
                <h3>Tags</h3>
                <div class="tags-list">
                  <span v-for="tag in selectedPersonality.tags" :key="tag" class="tag">
                    {{ tag }}
                  </span>
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.dependencies?.length">
                <h3>Dependencies</h3>
                <div class="dependencies-list">
                  <div v-for="dep in selectedPersonality.dependencies" :key="dep" class="dependency">
                    {{ dep }}
                  </div>
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.anti_prompts?.length">
                <h3>Anti-prompts</h3>
                <div class="anti-prompts-list">
                  <div v-for="prompt in selectedPersonality.anti_prompts" :key="prompt" class="anti-prompt">
                    "{{ prompt }}"
                  </div>
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.prompts_list?.length">
                <h3>Example Prompts</h3>
                <div class="prompts-list">
                  <div v-for="(prompt, index) in selectedPersonality.prompts_list.slice(0, 5)" :key="index" class="prompt-item">
                    <button @click="copyPrompt(prompt)" class="prompt-button">
                      {{ prompt.length > 100 ? prompt.substring(0, 100) + '...' : prompt }}
                    </button>
                  </div>
                  <div v-if="selectedPersonality.prompts_list.length > 5" class="more-prompts">
                    ... and {{ selectedPersonality.prompts_list.length - 5 }} more prompts
                  </div>
                </div>
              </div>

              <div class="detail-section" v-if="selectedPersonality.disclaimer">
                <h3>Disclaimer</h3>
                <div class="disclaimer-text">
                  {{ selectedPersonality.disclaimer }}
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
const selectedPersonality = ref<any>(null)

onMounted(() => {
  if (store.personalities.length === 0) {
    store.loadPersonalities()
  }
})

function selectPersonality(personality: any) {
  selectedPersonality.value = personality
}

function copyPrompt(prompt: string) {
  navigator.clipboard.writeText(prompt).then(() => {
    // Could add a toast notification here
    console.log('Prompt copied to clipboard')
  })
}
</script>

<style scoped>
.personalities {
  background: #1a237e;
  color: white;
}

.personalities-layout {
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

.personality-item {
  padding: 12px;
  border: 1px solid #1565c0;
  margin-bottom: 8px;
  cursor: pointer;
  transition: background-color 0.2s;
}

.personality-item:hover {
  background: #1565c0;
}

.personality-item.active {
  background: #1976d2;
  border-color: #2196f3;
}

.personality-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  margin-bottom: 4px;
}

.personality-name {
  font-weight: bold;
}

.personality-version {
  font-size: 12px;
  color: #bbbbbb;
}

.personality-author {
  font-size: 12px;
  color: #bbbbbb;
  margin-bottom: 4px;
}

.personality-category {
  font-size: 12px;
  background: #1976d2;
  padding: 2px 6px;
  display: inline-block;
  margin-bottom: 8px;
}

.personality-description {
  font-size: 12px;
  color: #cccccc;
  margin-bottom: 8px;
  line-height: 1.4;
}

.personality-tags {
  display: flex;
  gap: 4px;
  flex-wrap: wrap;
}

.tag {
  background: #424242;
  padding: 2px 6px;
  font-size: 10px;
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

.description-text, .conditioning-text, .welcome-text, .disclaimer-text {
  background: rgba(25, 118, 210, 0.1);
  border: 1px solid #1976d2;
  padding: 12px;
  line-height: 1.5;
  white-space: pre-wrap;
}

.tags-list {
  display: flex;
  gap: 8px;
  flex-wrap: wrap;
}

.dependencies-list, .anti-prompts-list {
  display: flex;
  flex-direction: column;
  gap: 4px;
}

.dependency, .anti-prompt {
  background: rgba(66, 66, 66, 0.5);
  padding: 4px 8px;
  font-family: monospace;
  font-size: 12px;
}

.prompts-list {
  display: flex;
  flex-direction: column;
  gap: 8px;
}

.prompt-button {
  background: #424242;
  border: 1px solid #666666;
  color: white;
  padding: 8px 12px;
  cursor: pointer;
  text-align: left;
  font-size: 12px;
  line-height: 1.4;
  transition: background-color 0.2s;
}

.prompt-button:hover {
  background: #555555;
}

.more-prompts {
  font-size: 12px;
  color: #bbbbbb;
  font-style: italic;
}

.loading, .empty {
  text-align: center;
  color: #bbbbbb;
  padding: 16px;
}
</style>

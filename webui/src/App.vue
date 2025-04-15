<script setup lang="ts">
import { RouterLink, RouterView } from 'vue-router'
// import HelloWorld from './components/HelloWorld.vue' // Default component
</script>

<template>
  <header>
    <!-- Optional Logo/Navigation -->
    <img alt="Vue logo" class="logo" src="@/assets/logo.svg" width="50" height="50" />
    <div class="wrapper">
      <!-- <HelloWorld msg="You did it!" /> -->
      <nav>
        <RouterLink to="/">Home</RouterLink>
        <RouterLink to="/chat">Chat</RouterLink>
        <!-- <RouterLink to="/about">About</RouterLink> -->
      </nav>
    </div>
  </header>

  <main>
     <RouterView /> <!-- Where routed components will be displayed -->
  </main>

  <footer>
    <!-- Optional Footer -->
    LOLLMS Web UI Placeholder
  </footer>
</template>

<style scoped>
  header {
    line-height: 1.5;
    max-height: 100vh;
    padding-bottom: 1rem;
    border-bottom: 1px solid var(--color-border);
    margin-bottom: 1rem;
  }
  .logo { display: block; margin: 0 auto 1rem; }
  nav { width: 100%; font-size: 12px; text-align: center; margin-top: 1rem;}
  nav a.router-link-exact-active { color: var(--color-text); }
  nav a.router-link-exact-active:hover { background-color: transparent; }
  nav a { display: inline-block; padding: 0 1rem; border-left: 1px solid var(--color-border); }
  nav a:first-of-type { border: 0; }
  main { margin-top: 2rem; }
  footer { margin-top: 2rem; text-align: center; font-size: 0.8em; color: #aaa;}

  /* Basic layout styles */
   header .wrapper { display: flex; place-items: flex-start; flex-wrap: wrap; }
   nav { margin-left: -1rem; font-size: 1rem; padding: 1rem 0; margin-top: 1rem; }

</style>
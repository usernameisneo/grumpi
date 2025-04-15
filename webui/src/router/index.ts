// webui/src/router/index.ts
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/HomeView.vue'
import ChatView from '../views/ChatView.vue' // Create this component later

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL), // Use browser history mode
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView // Default Vite home page for now
    },
    {
      path: '/chat',
      name: 'chat',
      component: ChatView // Route for the main chat interface
    },
    // Add other routes like /settings, /models etc. later
    // {
    //   path: '/about',
    //   name: 'about',
    //   // route level code-splitting
    //   // this generates a separate chunk (About.[hash].js) for this route
    //   // which is lazy-loaded when the route is visited.
    //   component: () => import('../views/AboutView.vue')
    // }
  ]
})

export default router
import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/home/home-page.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },
    {
      path: '/guide',
      name: 'about',
      component: () => import('../views/guide/guide-page.vue')
    }
  ]
})

export default router

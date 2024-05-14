import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/home/HomePage.vue'

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
      component: () => import('../views/guide/guidePage.vue')
    }
  ]
})

export default router

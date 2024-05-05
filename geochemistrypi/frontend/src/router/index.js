import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/Home/HomePage.vue'

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
      component: () => import('../views/Guide/GuidePage.vue')
    }
  ]
})

export default router

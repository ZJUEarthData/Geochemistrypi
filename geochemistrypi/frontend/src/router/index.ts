import { createRouter, createWebHistory } from 'vue-router'
import HomeView from '../views/home/home-page.vue'

const router = createRouter({
  history: createWebHistory(import.meta.env.BASE_URL),
  routes: [
    {
      path: '/',
      name: 'home',
      component: HomeView
    },{
      path: '/guide',
      name: 'guide',
      component: () => import('../views/guide/guide-index.vue')
    },
    {
      path: '/guide-view',
      name: 'guideView',
      component: () => import('../views/guide-view/guide-page.vue'),
      meta:{
        noNav:true
      }
    }
  ]
})

export default router

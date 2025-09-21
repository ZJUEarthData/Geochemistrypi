<script lang="ts" setup>
import { RouterView } from 'vue-router'
import NavBar from './components/nav-bar.vue'
</script>

<template>
  <header v-if="!$route.meta.noNav">
    <NavBar />
  </header>

  <RouterView />
</template>

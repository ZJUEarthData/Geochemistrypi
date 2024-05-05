<script setup>
import { RouterView } from 'vue-router'
import NavBar from './components/NavBar.vue'
</script>

<template>
  <header>
    <!-- <nav>
        <RouterLink to="/">Home</RouterLink>
      </nav> -->
    <NavBar />
  </header>

  <RouterView />
</template>

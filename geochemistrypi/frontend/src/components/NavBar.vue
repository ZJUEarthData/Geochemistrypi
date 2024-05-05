<script setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <nav class="nav-bar">
    <div class="logo"><img src="@/assets/imgs/Geochemistry_π.png"></div>
    <div class="nav-link">
      <RouterLink to="/">1111</RouterLink>

    </div>
  </nav>
</template>
<style lang="scss" scoped>
.nav-bar {
  display: flex;
  background: #000;
  height: 100px;
  align-items: center;
  justify-content: space-between;

  .logo {
    img {
      height: 85px;
    }
  }

  .nav-link {
    display: flex;
  }
}
</style>
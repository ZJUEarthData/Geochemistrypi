<script lang="ts" setup>
import { RouterLink } from 'vue-router'
</script>

<template>
  <nav class="nav-bar">
    <div class="logo"><img src="@/assets/imgs/logo.png" /></div>
    <div class="nav-link">
      <a href="https://github.com/ZJUEarthData/geochemistrypi/" target="_blank">Github</a>
      <a href="https://geochemistrypi.readthedocs.io/en/latest/" target="_blank">Docs</a>
      <RouterLink to="/guide">Guide</RouterLink>
      <RouterLink to="/"><span class="link-start">Get start</span></RouterLink>
      <RouterLink to="/">Sign in</RouterLink>
    </div>
  </nav>
</template>
<style lang="scss" scoped>
.nav-bar {
  display: flex;
  background: #000;
  height: 60px;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;
  .logo {
    img {
      height: 60px;
    }
  }
  .nav-link {
    display: flex;
    a {
      color: #fff;
      margin-right: 45px;
      line-height: 36px;
      font-size: 24px;
      &:last-of-type {
        margin-right: 0px;
      }
      &:hover {
        transform: scale(1.05);
      }
      .link-start {
        background-color: #f25b28;
        padding: 0 8px;
        height: 36px;
        line-height: 32px;
        border-radius: 4px;
        display: inline-block;
      }
    }
  }
}
</style>

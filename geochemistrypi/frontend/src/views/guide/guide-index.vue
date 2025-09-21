<template>
    <!-- <iframe height="100%" width="100%"  src="/guide-view" /> -->
    <GuideView />
</template>
<script setup lang="ts">
import GuideView from '../guide-view/guide-page.vue'
</script>
<style lang="scss" scoped>
iframe{
    width: 100%;
    height: calc(100vh - 66px);
}
</style>
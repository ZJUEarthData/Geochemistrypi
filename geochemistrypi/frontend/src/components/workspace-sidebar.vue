<script lang="ts" setup>
import { RouterLink } from 'vue-router'

import { t } from '@/i18n'

defineProps<{
  active: 'online' | 'data-mining'
}>()
</script>

<template>
  <aside class="workspace-sidebar">
    <nav :aria-label="t('Analysis modules', '分析模块')">
      <RouterLink class="workspace-link" :class="{ active: active === 'online' }" to="/online">
        <el-icon><DataAnalysis /></el-icon>
        <span>{{ t('Chemical modeling', '化学建模') }}</span>
      </RouterLink>
      <RouterLink
        class="workspace-link"
        :class="{ active: active === 'data-mining' }"
        to="/data-mining"
      >
        <el-icon><Files /></el-icon>
        <span>{{ t('Data mining', '数据挖掘') }}</span>
      </RouterLink>
    </nav>

    <figure class="field-figure">
      <img
        src="@/assets/imgs/alpine-field-photo.jpg"
        :alt="t(
          'Snow-covered mountain above a green meadow and clear alpine stream',
          '雪山、绿色草甸与清澈高山溪流'
        )"
      />
    </figure>
  </aside>
</template>

<style lang="scss" scoped>
.workspace-sidebar {
  display: flex;
  position: sticky;
  top: 72px;
  flex-direction: column;
  align-self: start;
  height: calc(100vh - 72px);
  overflow: hidden;
  border-right: 1px solid #d7e7e4;
  background: #f8fcfb;

  nav {
    display: grid;
    flex: 0 0 auto;
    padding: 24px 0 100px;
  }
}

.workspace-link {
  display: flex;
  position: relative;
  align-items: center;
  gap: 12px;
  min-height: 58px;
  padding: 0 22px;
  color: #607b7d;
  font-size: 14px;
  font-weight: 580;
  transition:
    color 0.2s ease,
    background-color 0.2s ease;

  &::before {
    position: absolute;
    top: 12px;
    bottom: 12px;
    left: 0;
    width: 3px;
    background: #e56f55;
    content: '';
    opacity: 0;
  }

  .el-icon {
    color: #4f9aa0;
    font-size: 21px;
  }

  &:hover,
  &:focus-visible,
  &.active {
    color: #173f47;
    background: #eaf5f2;
    outline: none;
  }

  &.active::before {
    opacity: 1;
  }

  &.active .el-icon {
    color: #d86149;
  }
}

.field-figure {
  position: relative;
  flex: 1 1 auto;
  min-height: 380px;
  margin: 0;
  overflow: hidden;
  border-top: 1px solid #d7e7e4;
  background: #dceceb;

  img {
    width: 100%;
    height: 100%;
    object-fit: cover;
    object-position: center 42%;
  }

}

@media (max-width: 1360px) {
  .workspace-sidebar nav {
    padding-bottom: 78px;
  }

  .workspace-link {
    gap: 10px;
    padding-right: 17px;
    padding-left: 17px;
    font-size: 13.5px;
  }
}

@media (max-width: 820px) {
  .workspace-sidebar {
    position: static;
    height: auto;
    border-right: 0;
    border-bottom: 1px solid #d7e7e4;

    nav {
      grid-template-columns: 1fr 1fr;
      padding: 0;
    }
  }

  .workspace-link {
    justify-content: center;
    min-height: 60px;
    padding: 0 12px;

    &::before {
      top: auto;
      right: 24px;
      bottom: 0;
      left: 24px;
      width: auto;
      height: 3px;
    }
  }

  .field-figure {
    display: none;
  }
}
</style>

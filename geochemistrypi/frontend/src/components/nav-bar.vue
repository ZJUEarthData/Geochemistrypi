<script lang="ts" setup>
import { ref } from 'vue'
import { RouterLink } from 'vue-router'

import { authUiEnabled } from '@/config/features'
import { locale, setLocale, t } from '@/i18n'

const moreMenu = ref<HTMLDetailsElement | null>(null)

function toggleLocale() {
  setLocale(locale.value === 'en' ? 'zh' : 'en')
}

function closeMoreMenu() {
  if (moreMenu.value) moreMenu.value.open = false
}
</script>

<template>
  <nav class="nav-bar" :aria-label="t('Primary navigation', '主导航')">
    <div class="brand-tools">
      <RouterLink class="logo" to="/">
        <img class="logo-mark" src="@/assets/imgs/onlyLogo.png" alt="" />
        <span class="logo-copy">
          <strong>Geochemistry π</strong>
          <small>{{
            t('A PYTHON FRAMEWORK FOR GEOCHEMISTRY', '面向地球化学的 PYTHON 框架')
          }}</small>
        </span>
      </RouterLink>
      <button
        class="language-toggle"
        type="button"
        :aria-label="t('Switch to Chinese', '切换为英文')"
        @click="toggleLocale"
      >
        <span>{{ locale === 'en' ? 'EN / 中文' : '中 / English' }}</span>
        <el-icon aria-hidden="true"><ArrowDown /></el-icon>
      </button>
    </div>

    <div class="module-links">
      <RouterLink to="/online">{{ t('Chemical modeling', '化学建模') }}</RouterLink>
      <RouterLink to="/data-mining">{{ t('Data mining', '数据挖掘') }}</RouterLink>
    </div>

    <div class="utility-links">
      <a
        href="https://geochemistrypi.readthedocs.io/en/latest/"
        target="_blank"
        rel="noopener noreferrer"
      >
        {{ t('Docs', '文档') }}
      </a>
      <RouterLink to="/guide">{{ t('About us', '关于我们') }}</RouterLink>
      <RouterLink v-if="authUiEnabled" class="account-link" to="/login">
        {{ t('Login / Register', '登录 / 注册') }}
      </RouterLink>
      <RouterLink class="icon-link" to="/" :aria-label="t('Search', '搜索')">
        <el-icon><Search /></el-icon>
      </RouterLink>
      <details ref="moreMenu" class="more-menu">
        <summary :aria-label="t('Open navigation menu', '打开导航菜单')">
          <el-icon><Menu /></el-icon>
        </summary>
        <div class="more-menu-panel">
          <a
            href="https://geochemistrypi.readthedocs.io/en/latest/"
            target="_blank"
            rel="noopener noreferrer"
            @click="closeMoreMenu"
          >
            {{ t('Documentation', '使用文档') }}
          </a>
          <RouterLink to="/guide" @click="closeMoreMenu">
            {{ t('About us', '关于我们') }}
          </RouterLink>
          <a
            href="https://github.com/ZJUEarthData/geochemistrypi/"
            target="_blank"
            rel="noopener noreferrer"
            @click="closeMoreMenu"
          >
            GitHub
          </a>
          <RouterLink
            v-if="authUiEnabled"
            class="account-menu-link"
            to="/login"
            @click="closeMoreMenu"
          >
            {{ t('Login / Register', '登录 / 注册') }}
          </RouterLink>
        </div>
      </details>
    </div>
  </nav>
</template>

<style lang="scss" scoped>
.nav-bar {
  --nav-border: rgb(153 212 220 / 22%);
  display: grid;
  position: sticky;
  z-index: 100;
  top: 0;
  grid-template-columns: minmax(280px, 1fr) auto minmax(280px, 1fr);
  align-items: center;
  min-height: 72px;
  padding: 0 28px;
  border-bottom: 1px solid var(--nav-border);
  color: #eef8f8;
  background: #113a46;
  box-shadow: 0 8px 24px rgb(17 58 70 / 10%);
}

.brand-tools,
.module-links,
.utility-links {
  display: flex;
  align-items: center;
}

.brand-tools {
  gap: 22px;
  min-width: 0;
}

.logo {
  display: flex;
  align-items: center;
  gap: 10px;
  min-width: 0;

  .logo-mark {
    width: 48px;
    height: 48px;
    object-fit: contain;
  }

  .logo-copy {
    display: grid;
    gap: 2px;

    strong {
      color: #f4fafa;
      line-height: 1;
      font-size: 18px;
      font-weight: 720;
      letter-spacing: -0.025em;
    }

    small {
      color: #b8d5d4;
      font-size: 6.5px;
      font-weight: 650;
      letter-spacing: 0.09em;
      white-space: nowrap;
    }
  }
}

.language-toggle {
  display: inline-flex;
  align-items: center;
  gap: 8px;
  height: 38px;
  padding: 0 15px;
  border: 1px solid rgb(196 229 232 / 34%);
  border-radius: 999px;
  color: #d9eff1;
  background: rgb(255 255 255 / 5%);
  font: inherit;
  font-size: 14px;
  cursor: pointer;
  transition:
    border-color 0.2s ease,
    color 0.2s ease,
    background-color 0.2s ease;

  &:hover,
  &:focus-visible {
    border-color: #78cdd2;
    color: #fff;
    background: rgb(120 205 210 / 9%);
    outline: none;
  }

  .el-icon {
    font-size: 12px;
  }
}

.module-links {
  align-self: stretch;
  gap: 34px;

  a {
    display: flex;
    position: relative;
    align-items: center;
    padding: 0 4px;
    color: #a9c4ce;
    font-size: 16px;
    white-space: nowrap;
    transition: color 0.2s ease;

    &::after {
      position: absolute;
      right: 0;
      bottom: 0;
      left: 0;
      height: 3px;
      background: #ee6b52;
      content: '';
      transform: scaleX(0);
      transform-origin: center;
      transition: transform 0.2s ease;
    }

    &:hover,
    &:focus-visible,
    &.router-link-active {
      color: #fff;
      outline: none;
    }

    &.router-link-active::after {
      transform: scaleX(1);
    }
  }
}

.utility-links {
  justify-content: flex-end;
  gap: 28px;

  > a {
    color: #bdd2d8;
    font-size: 14px;
    white-space: nowrap;
    transition: color 0.2s ease;

    &:hover,
    &:focus-visible {
      color: #fff;
      outline: none;
    }
  }

  .icon-link {
    display: inline-flex;
    padding-left: 18px;
    border-left: 1px solid var(--nav-border);
    color: #c9e4e7;
    font-size: 23px;
  }

  .account-link.router-link-active {
    color: #fff;
  }
}

.more-menu {
  display: none;
  position: relative;

  summary {
    display: inline-flex;
    align-items: center;
    justify-content: center;
    width: 38px;
    height: 38px;
    border: 1px solid rgb(196 229 232 / 28%);
    border-radius: 50%;
    color: #d9eff1;
    cursor: pointer;
    list-style: none;

    &::-webkit-details-marker {
      display: none;
    }
  }
}

.more-menu-panel {
  display: grid;
  position: absolute;
  top: calc(100% + 12px);
  right: 0;
  min-width: 188px;
  overflow: hidden;
  border: 1px solid #355d70;
  border-radius: 10px;
  background: #0b2b40;
  box-shadow: 0 18px 40px rgb(1 14 24 / 34%);

  a {
    padding: 12px 15px;
    color: #d9eff1;
    font-size: 14px;

    &:hover,
    &:focus-visible {
      color: #fff;
      background: #143b50;
      outline: none;
    }
  }

  .account-menu-link {
    border-top: 1px solid #355d70;
  }
}

@media (max-width: 1120px) {
  .nav-bar {
    grid-template-columns: 1fr auto;
    grid-template-rows: 72px 52px;
    padding: 0 20px;
  }

  .brand-tools {
    grid-row: 1;
    grid-column: 1;
  }

  .module-links {
    display: flex;
    grid-row: 2;
    grid-column: 1 / -1;
    align-self: stretch;
    gap: 30px;
    margin: 0 -20px;
    padding: 0 20px;
    overflow-x: auto;
    border-top: 1px solid var(--nav-border);
    background: #123e4a;
    scrollbar-width: none;

    &::-webkit-scrollbar {
      display: none;
    }

    a {
      flex: 0 0 auto;
      min-height: 52px;
    }
  }

  .utility-links {
    grid-row: 1;
    grid-column: 2;
  }

  .utility-links > a:not(.icon-link) {
    display: none;
  }

  .more-menu {
    display: block;
  }
}

@media (max-width: 640px) {
  .nav-bar {
    min-height: 64px;
    grid-template-rows: 64px 50px;
    padding: 0 14px;
  }

  .brand-tools {
    gap: 10px;
  }

  .logo .logo-mark {
    width: 42px;
    height: 42px;
  }

  .logo .logo-copy {
    strong {
      font-size: 15px;
    }

    small {
      display: none;
    }
  }

  .language-toggle {
    width: 38px;
    padding: 0;
    justify-content: center;

    span {
      display: none;
    }
  }

  .utility-links {
    gap: 10px;
  }

  .module-links {
    gap: 24px;
    margin: 0 -14px;
    padding: 0 20px;

    a {
      min-height: 50px;
      font-size: 15px;
    }
  }
}

@media (max-width: 480px) {
  .utility-links .icon-link {
    display: none;
  }
}
</style>

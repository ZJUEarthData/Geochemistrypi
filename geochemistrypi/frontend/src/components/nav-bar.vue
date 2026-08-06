<script lang="ts" setup>
import { ref } from 'vue'
import { RouterLink } from 'vue-router'

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
  <nav class="nav-bar">
    <div class="brand-tools">
      <RouterLink class="logo" to="/">
        <img src="@/assets/imgs/logo.png" :alt="t('Geochemistry Pi', '地球化学π')" />
      </RouterLink>
      <button
        class="language-toggle"
        type="button"
        :aria-label="t('Switch to Chinese', '切换为英文')"
        @click="toggleLocale"
      >
        <span class="language-icon" aria-hidden="true">A/文</span>
        <span>{{ locale === 'en' ? '中文' : 'English' }}</span>
      </button>
    </div>
    <div class="nav-link">
      <div class="module-links">
        <RouterLink to="/online">
          <span class="link-start">{{ t('Chemical modeling', '化学建模') }}</span>
        </RouterLink>
        <RouterLink to="/data-mining">
          <span class="link-start">{{ t('Data mining', '数据挖掘') }}</span>
        </RouterLink>
      </div>
      <details ref="moreMenu" class="more-menu">
        <summary>{{ t('More', '更多') }}</summary>
        <div class="more-menu-panel">
          <a
            href="https://github.com/ZJUEarthData/geochemistrypi/"
            target="_blank"
            rel="noopener noreferrer"
            @click="closeMoreMenu"
          >
            GitHub
          </a>
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
          <RouterLink to="/" @click="closeMoreMenu">
            {{ t('Search', '搜索') }}
          </RouterLink>
          <RouterLink to="/" @click="closeMoreMenu">
            {{ t('Account', '用户中心') }}
          </RouterLink>
        </div>
      </details>
      <a href="https://github.com/ZJUEarthData/geochemistrypi/" target="_blank"
        ><i
          class="el-icon"
          style="font-size: 32px; vertical-align: middle; margin-right: 8px"
          data-v-72eabb9c=""
          ><!--[--><svg
            preserveAspectRatio="xMidYMid meet"
            viewBox="0 0 24 24"
            width="1.2em"
            height="1.2em"
            data-v-72eabb9c=""
          >
            <path
              fill="currentColor"
              d="M12 2C6.475 2 2 6.475 2 12a9.994 9.994 0 0 0 6.838 9.488c.5.087.687-.213.687-.476c0-.237-.013-1.024-.013-1.862c-2.512.463-3.162-.612-3.362-1.175c-.113-.288-.6-1.175-1.025-1.413c-.35-.187-.85-.65-.013-.662c.788-.013 1.35.725 1.538 1.025c.9 1.512 2.338 1.087 2.912.825c.088-.65.35-1.087.638-1.337c-2.225-.25-4.55-1.113-4.55-4.938c0-1.088.387-1.987 1.025-2.688c-.1-.25-.45-1.275.1-2.65c0 0 .837-.262 2.75 1.026a9.28 9.28 0 0 1 2.5-.338c.85 0 1.7.112 2.5.337c1.912-1.3 2.75-1.024 2.75-1.024c.55 1.375.2 2.4.1 2.65c.637.7 1.025 1.587 1.025 2.687c0 3.838-2.337 4.688-4.562 4.938c.362.312.675.912.675 1.85c0 1.337-.013 2.412-.013 2.75c0 .262.188.574.688.474A10.016 10.016 0 0 0 22 12c0-5.525-4.475-10-10-10z"
            ></path></svg
          ><!--]--></i
        >Github</a
      >
      <a href="https://geochemistrypi.readthedocs.io/en/latest/" target="_blank">Docs</a>
      <RouterLink to="/guide">{{ t('About us', '关于我们') }}</RouterLink>
      <!-- <RouterLink to="/"><span class="link-start">Get started</span></RouterLink>
      <RouterLink to="/">Sign in</RouterLink> -->
      <RouterLink to="/"
        ><svg
          width="30"
          height="30"
          viewBox="0 0 30 30"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M27.6667 30L17.1667 19.5C16.3333 20.1667 15.375 20.6944 14.2917 21.0833C13.2083 21.4722 12.0556 21.6667 10.8333 21.6667C7.80556 21.6667 5.24306 20.6181 3.14583 18.5208C1.04861 16.4236 0 13.8611 0 10.8333C0 7.80556 1.04861 5.24306 3.14583 3.14583C5.24306 1.04861 7.80556 0 10.8333 0C13.8611 0 16.4236 1.04861 18.5208 3.14583C20.6181 5.24306 21.6667 7.80556 21.6667 10.8333C21.6667 12.0556 21.4722 13.2083 21.0833 14.2917C20.6944 15.375 20.1667 16.3333 19.5 17.1667L30 27.6667L27.6667 30ZM10.8333 18.3333C12.9167 18.3333 14.6875 17.6042 16.1458 16.1458C17.6042 14.6875 18.3333 12.9167 18.3333 10.8333C18.3333 8.75 17.6042 6.97917 16.1458 5.52083C14.6875 4.0625 12.9167 3.33333 10.8333 3.33333C8.75 3.33333 6.97917 4.0625 5.52083 5.52083C4.0625 6.97917 3.33333 8.75 3.33333 10.8333C3.33333 12.9167 4.0625 14.6875 5.52083 16.1458C6.97917 17.6042 8.75 18.3333 10.8333 18.3333Z"
            fill="#F25B28"
          />
        </svg>
      </RouterLink>
      <RouterLink to="/"
        ><svg
          width="30"
          height="30"
          viewBox="0 0 30 30"
          fill="none"
          xmlns="http://www.w3.org/2000/svg"
        >
          <path
            d="M10.5 18.375C9.975 18.375 9.53125 18.1938 9.16875 17.8312C8.80625 17.4688 8.625 17.025 8.625 16.5C8.625 15.975 8.80625 15.5312 9.16875 15.1688C9.53125 14.8063 9.975 14.625 10.5 14.625C11.025 14.625 11.4688 14.8063 11.8313 15.1688C12.1938 15.5312 12.375 15.975 12.375 16.5C12.375 17.025 12.1938 17.4688 11.8313 17.8312C11.4688 18.1938 11.025 18.375 10.5 18.375ZM19.5 18.375C18.975 18.375 18.5312 18.1938 18.1688 17.8312C17.8062 17.4688 17.625 17.025 17.625 16.5C17.625 15.975 17.8062 15.5312 18.1688 15.1688C18.5312 14.8063 18.975 14.625 19.5 14.625C20.025 14.625 20.4688 14.8063 20.8312 15.1688C21.1938 15.5312 21.375 15.975 21.375 16.5C21.375 17.025 21.1938 17.4688 20.8312 17.8312C20.4688 18.1938 20.025 18.375 19.5 18.375ZM15 27C18.35 27 21.1875 25.8375 23.5125 23.5125C25.8375 21.1875 27 18.35 27 15C27 14.4 26.9625 13.8188 26.8875 13.2563C26.8125 12.6938 26.675 12.15 26.475 11.625C25.95 11.75 25.425 11.8438 24.9 11.9062C24.375 11.9688 23.825 12 23.25 12C20.975 12 18.825 11.5125 16.8 10.5375C14.775 9.5625 13.05 8.2 11.625 6.45C10.825 8.4 9.68125 10.0938 8.19375 11.5312C6.70625 12.9688 4.975 14.05 3 14.775V15C3 18.35 4.1625 21.1875 6.4875 23.5125C8.8125 25.8375 11.65 27 15 27ZM15 30C12.925 30 10.975 29.6062 9.15 28.8187C7.325 28.0312 5.7375 26.9625 4.3875 25.6125C3.0375 24.2625 1.96875 22.675 1.18125 20.85C0.39375 19.025 0 17.075 0 15C0 12.925 0.39375 10.975 1.18125 9.15C1.96875 7.325 3.0375 5.7375 4.3875 4.3875C5.7375 3.0375 7.325 1.96875 9.15 1.18125C10.975 0.39375 12.925 0 15 0C17.075 0 19.025 0.39375 20.85 1.18125C22.675 1.96875 24.2625 3.0375 25.6125 4.3875C26.9625 5.7375 28.0312 7.325 28.8187 9.15C29.6062 10.975 30 12.925 30 15C30 17.075 29.6062 19.025 28.8187 20.85C28.0312 22.675 26.9625 24.2625 25.6125 25.6125C24.2625 26.9625 22.675 28.0312 20.85 28.8187C19.025 29.6062 17.075 30 15 30ZM12.975 3.1875C14.025 4.9375 15.45 6.34375 17.25 7.40625C19.05 8.46875 21.05 9 23.25 9C23.6 9 23.9375 8.98125 24.2625 8.94375C24.5875 8.90625 24.925 8.8625 25.275 8.8125C24.225 7.0625 22.8 5.65625 21 4.59375C19.2 3.53125 17.2 3 15 3C14.65 3 14.3125 3.01875 13.9875 3.05625C13.6625 3.09375 13.325 3.1375 12.975 3.1875ZM3.6375 11.2125C4.9125 10.4875 6.025 9.55 6.975 8.4C7.925 7.25 8.6375 5.9625 9.1125 4.5375C7.8375 5.2625 6.725 6.2 5.775 7.35C4.825 8.5 4.1125 9.7875 3.6375 11.2125Z"
            fill="#F25B28"
          />
        </svg>
      </RouterLink>
    </div>
  </nav>
</template>
<style lang="scss" scoped>
.nav-bar {
  display: flex;
  position: relative;
  z-index: 100;
  background: #000;
  height: 60px;
  align-items: center;
  justify-content: space-between;
  padding: 0 20px;

  .brand-tools {
    display: flex;
    align-items: center;
    gap: 30px;
    flex-shrink: 0;
  }

  .logo {
    display: flex;

    img {
      height: 60px;
    }
  }

  .language-toggle {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    min-width: 112px;
    height: 34px;
    padding: 0 13px;
    border: 1px solid #707070;
    border-radius: 999px;
    color: #fff;
    background: #171717;
    font: inherit;
    font-size: 14px;
    cursor: pointer;
    transition:
      border-color 0.2s ease,
      background-color 0.2s ease;

    &:hover,
    &:focus-visible {
      border-color: #f25b28;
      background: #292929;
      outline: none;
    }

    .language-icon {
      color: #f25b28;
      font-weight: 700;
    }
  }

  .nav-link {
    display: flex;
    align-items: center;

    .more-menu {
      display: none;
      position: relative;

      summary {
        height: 36px;
        padding: 0 13px;
        border: 1px solid #f25b28;
        border-radius: 4px;
        color: #fff;
        line-height: 34px;
        font-size: 15px;
        cursor: pointer;
        list-style: none;
        white-space: nowrap;

        &::-webkit-details-marker {
          display: none;
        }

        &::after {
          margin-left: 7px;
          content: '▾';
        }
      }

      &[open] summary {
        background: #f25b28;
      }

      .more-menu-panel {
        display: grid;
        position: absolute;
        top: calc(100% + 8px);
        right: 0;
        min-width: 190px;
        overflow: hidden;
        border: 1px solid #374151;
        border-radius: 8px;
        background: #111827;
        box-shadow: 0 12px 28px rgb(0 0 0 / 28%);

        a {
          margin: 0;
          padding: 11px 14px;
          color: #fff;
          line-height: 1.25;
          font-size: 14px;
          white-space: nowrap;

          &:hover,
          &:focus-visible {
            background: #263244;
            color: #ff8b63;
            outline: none;
            transform: none;
          }
        }
      }
    }

    .module-links {
      display: flex;
      gap: 10px;
      margin-right: 45px;

      a {
        margin-right: 0;
      }
    }

    a {
      color: #fff;
      margin-right: 45px;
      line-height: 36px;
      font-size: 20px;
      display: flex;
      align-items: center;

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
        white-space: nowrap;
      }
    }
  }
}

@media (max-width: 1240px) {
  .nav-bar {
    padding: 0 14px;

    .brand-tools {
      gap: 14px;
    }

    .logo img {
      width: 150px;
      object-fit: contain;
    }

    .language-toggle {
      min-width: 88px;
      padding: 0 10px;
    }

    .nav-link {
      gap: 8px;

      .more-menu {
        display: block;
      }

      .module-links {
        margin-right: 0;

        a {
          font-size: 16px;
        }
      }

      > a {
        display: none;
      }
    }
  }
}

@media (max-width: 900px) {
  .nav-bar {
    height: auto;
    min-height: 88px;
    flex-wrap: wrap;
    align-content: center;
    padding: 4px 10px 8px;

    .brand-tools {
      width: 100%;
      gap: 10px;
    }

    .logo img {
      width: 104px;
      height: 42px;
      object-fit: contain;
    }

    .language-toggle {
      min-width: 48px;
      height: 30px;
      padding: 0 8px;
      font-size: 12px;

      .language-icon {
        display: none;
      }
    }

    .nav-link {
      width: 100%;
      gap: 4px;

      .module-links {
        display: grid;
        grid-template-columns: 1fr 1fr;
        width: 100%;
        gap: 4px;
        margin-right: 0;

        a {
          justify-content: stretch;
          font-size: 12px;

          .link-start {
            height: 30px;
            width: 100%;
            max-width: none;
            padding: 0 5px;
            line-height: 28px;
            text-align: center;
            overflow: hidden;
            text-overflow: ellipsis;
            white-space: nowrap;
          }
        }
      }

      .more-menu {
        flex: 0 0 auto;

        summary {
          height: 30px;
          padding: 0 8px;
          line-height: 28px;
          font-size: 12px;

          &::after {
            margin-left: 4px;
          }
        }

        .more-menu-panel {
          top: calc(100% + 6px);
          min-width: 160px;
        }
      }
    }
  }
}
</style>

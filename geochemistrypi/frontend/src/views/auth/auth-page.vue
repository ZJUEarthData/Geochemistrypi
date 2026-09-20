<script lang="ts" setup>
import { computed, reactive, ref, watch } from 'vue'

import { t } from '@/i18n'

type Mode = 'login' | 'register'

const mode = ref<Mode>('login')
const notice = ref('')
const login = reactive({ identity: '', password: '' })
const registration = reactive({ username: '', email: '', password: '', confirmation: '' })

const passwordsMismatch = computed(
  () =>
    Boolean(registration.confirmation) && registration.password !== registration.confirmation
)

watch(mode, () => {
  notice.value = ''
})

function submitLogin() {
  notice.value = t(
    'Login interface ready. Account authentication is not connected yet.',
    '登录界面已完成，账户认证服务尚未连接。'
  )
}

function submitRegistration() {
  if (passwordsMismatch.value) return
  notice.value = t(
    'Registration interface ready. Account creation is not connected yet.',
    '注册界面已完成，账户创建服务尚未连接。'
  )
}

function showPasswordNotice() {
  notice.value = t(
    'Password recovery is not connected yet.',
    '密码找回服务尚未连接。'
  )
}
</script>

<template>
  <main class="auth-page">
    <section class="auth-shell" aria-labelledby="auth-title">
      <div class="auth-intro">
        <p class="eyebrow">GEOCHEMISTRY π ONLINE</p>
        <h1 id="auth-title">{{ t('Welcome to your workspace', '欢迎进入工作空间') }}</h1>
        <p>
          {{
            t(
              'Sign in to continue your work or create a new account. Online tools remain available without signing in during this interface preview.',
              '登录后可继续工作，或创建一个新账户。在当前界面预览阶段，不登录仍可使用 Online 功能。'
            )
          }}
        </p>
        <div class="preview-note">
          <span aria-hidden="true"></span>
          {{ t('Interface preview — no account service connected', '界面预览——尚未连接账户服务') }}
        </div>
      </div>

      <section class="auth-card" :aria-label="t('Login and registration', '登录和注册')">
        <div class="auth-tabs" role="tablist" :aria-label="t('Account form', '账户表单')">
          <button
            type="button"
            role="tab"
            :aria-selected="mode === 'login'"
            :class="{ active: mode === 'login' }"
            @click="mode = 'login'"
          >
            {{ t('Login', '登录') }}
          </button>
          <button
            type="button"
            role="tab"
            :aria-selected="mode === 'register'"
            :class="{ active: mode === 'register' }"
            @click="mode = 'register'"
          >
            {{ t('Register', '注册') }}
          </button>
        </div>

        <form v-if="mode === 'login'" class="auth-form" @submit.prevent="submitLogin">
          <div class="form-heading">
            <h2>{{ t('Welcome back', '欢迎回来') }}</h2>
            <p>{{ t('Enter your account details below.', '请在下方输入账户信息。') }}</p>
          </div>

          <label>
            <span>{{ t('Username or email', '用户名或邮箱') }}</span>
            <input
              v-model.trim="login.identity"
              name="identity"
              type="text"
              autocomplete="username"
              :placeholder="t('Enter username or email', '请输入用户名或邮箱')"
              required
            />
          </label>

          <label>
            <span>{{ t('Password', '密码') }}</span>
            <input
              v-model="login.password"
              name="password"
              type="password"
              autocomplete="current-password"
              :placeholder="t('Enter password', '请输入密码')"
              minlength="8"
              required
            />
          </label>

          <button class="forgot-button" type="button" @click="showPasswordNotice">
            {{ t('Forgot password?', '忘记密码？') }}
          </button>

          <button class="submit-button" type="submit">{{ t('Login', '登录') }}</button>
        </form>

        <form v-else class="auth-form" @submit.prevent="submitRegistration">
          <div class="form-heading">
            <h2>{{ t('Create an account', '创建账户') }}</h2>
            <p>{{ t('Complete the information below.', '请填写以下账户信息。') }}</p>
          </div>

          <label>
            <span>{{ t('Username', '用户名') }}</span>
            <input
              v-model.trim="registration.username"
              name="username"
              type="text"
              autocomplete="username"
              :placeholder="t('Choose a username', '请输入用户名')"
              minlength="3"
              required
            />
          </label>

          <label>
            <span>{{ t('Email', '邮箱') }}</span>
            <input
              v-model.trim="registration.email"
              name="email"
              type="email"
              autocomplete="email"
              :placeholder="t('Enter your email', '请输入邮箱')"
              required
            />
          </label>

          <label>
            <span>{{ t('Password', '密码') }}</span>
            <input
              v-model="registration.password"
              name="new-password"
              type="password"
              autocomplete="new-password"
              :placeholder="t('At least 8 characters', '至少输入 8 个字符')"
              minlength="8"
              required
            />
          </label>

          <label>
            <span>{{ t('Confirm password', '重复输入密码') }}</span>
            <input
              v-model="registration.confirmation"
              name="confirm-password"
              type="password"
              autocomplete="new-password"
              :aria-invalid="passwordsMismatch"
              :placeholder="t('Enter password again', '请再次输入密码')"
              minlength="8"
              required
            />
            <small v-if="passwordsMismatch" class="field-error">
              {{ t('The passwords do not match.', '两次输入的密码不一致。') }}
            </small>
          </label>

          <button class="submit-button" type="submit" :disabled="passwordsMismatch">
            {{ t('Create account', '创建账户') }}
          </button>
        </form>

        <p v-if="notice" class="form-notice" role="status">{{ notice }}</p>
      </section>
    </section>
  </main>
</template>

<style lang="scss" scoped>
.auth-page {
  display: grid;
  min-height: calc(100vh - 72px);
  padding: 56px 28px;
  background:
    radial-gradient(circle at 15% 15%, rgb(103 208 162 / 14%), transparent 32%),
    linear-gradient(135deg, #f5fbfa 0%, #eef7f6 55%, #f8fbfc 100%);
  color: #183e45;
  place-items: center;
}

.auth-shell {
  display: grid;
  grid-template-columns: minmax(0, 1fr) minmax(390px, 500px);
  gap: 72px;
  align-items: center;
  width: min(1080px, 100%);
}

.auth-intro {
  max-width: 520px;

  .eyebrow {
    margin-bottom: 14px;
    color: #d95f4b;
    font-size: 13px;
    font-weight: 750;
    letter-spacing: 0.16em;
  }

  h1 {
    margin: 0;
    color: #173f46;
    line-height: 1.08;
    font-size: clamp(38px, 5vw, 60px);
    font-weight: 760;
    letter-spacing: -0.045em;
  }

  > p:not(.eyebrow) {
    margin-top: 22px;
    color: #5d777c;
    line-height: 1.75;
    font-size: 16px;
  }
}

.preview-note {
  display: inline-flex;
  align-items: center;
  gap: 10px;
  margin-top: 28px;
  padding: 10px 14px;
  border: 1px solid #cfe5e1;
  border-radius: 999px;
  color: #456b70;
  background: rgb(255 255 255 / 68%);
  font-size: 13px;

  span {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: #67d0a2;
    box-shadow: 0 0 0 4px rgb(103 208 162 / 17%);
  }
}

.auth-card {
  padding: 34px;
  border: 1px solid #dbeae8;
  border-radius: 22px;
  background: rgb(255 255 255 / 94%);
  box-shadow: 0 24px 70px rgb(27 79 83 / 13%);
}

.auth-tabs {
  display: grid;
  grid-template-columns: 1fr 1fr;
  margin-bottom: 30px;
  padding: 4px;
  border-radius: 12px;
  background: #edf6f4;

  button {
    min-height: 42px;
    border: 0;
    border-radius: 9px;
    color: #688085;
    background: transparent;
    font: inherit;
    font-weight: 650;
    cursor: pointer;

    &.active {
      color: #173f46;
      background: #fff;
      box-shadow: 0 5px 16px rgb(27 79 83 / 10%);
    }

    &:focus-visible {
      outline: 2px solid #67b9bd;
      outline-offset: 2px;
    }
  }
}

.auth-form {
  display: grid;
  gap: 18px;
}

.form-heading {
  margin-bottom: 2px;

  h2 {
    margin: 0;
    color: #173f46;
    font-size: 26px;
    font-weight: 730;
  }

  p {
    margin-top: 4px;
    color: #789095;
  }
}

label {
  display: grid;
  gap: 7px;

  > span {
    color: #355e64;
    font-size: 13px;
    font-weight: 650;
  }
}

input {
  width: 100%;
  height: 48px;
  padding: 0 14px;
  border: 1px solid #cddfdd;
  border-radius: 10px;
  color: #173f46;
  background: #fbfdfd;
  font: inherit;
  transition:
    border-color 0.2s ease,
    box-shadow 0.2s ease;

  &::placeholder {
    color: #9aabad;
  }

  &:focus {
    border-color: #62b9b8;
    outline: none;
    box-shadow: 0 0 0 3px rgb(98 185 184 / 14%);
  }

  &[aria-invalid='true'] {
    border-color: #d95f4b;
  }
}

.forgot-button {
  justify-self: end;
  margin-top: -7px;
  padding: 0;
  border: 0;
  color: #c55240;
  background: transparent;
  font: inherit;
  font-size: 13px;
  cursor: pointer;

  &:hover,
  &:focus-visible {
    color: #9f3f31;
    text-decoration: underline;
    outline: none;
  }
}

.submit-button {
  height: 48px;
  margin-top: 3px;
  border: 0;
  border-radius: 10px;
  color: #fff;
  background: #d95f4b;
  font: inherit;
  font-weight: 700;
  cursor: pointer;
  transition:
    background-color 0.2s ease,
    transform 0.2s ease;

  &:hover:not(:disabled),
  &:focus-visible {
    background: #e56a54;
    outline: none;
    transform: translateY(-1px);
  }

  &:disabled {
    opacity: 0.48;
    cursor: not-allowed;
  }
}

.field-error {
  color: #c55240;
  font-size: 12px;
}

.form-notice {
  margin: 18px 0 0;
  padding: 11px 13px;
  border: 1px solid #cde5df;
  border-radius: 9px;
  color: #376c63;
  background: #eff8f5;
  line-height: 1.5;
  font-size: 13px;
}

@media (max-width: 900px) {
  .auth-page {
    padding: 40px 20px;
  }

  .auth-shell {
    grid-template-columns: 1fr;
    gap: 34px;
    width: min(560px, 100%);
  }

  .auth-intro {
    text-align: center;
  }

  .preview-note {
    justify-content: center;
  }
}

@media (max-width: 520px) {
  .auth-page {
    min-height: calc(100vh - 64px);
    padding: 28px 14px;
  }

  .auth-intro h1 {
    font-size: 36px;
  }

  .auth-card {
    padding: 24px 18px;
    border-radius: 16px;
  }
}
</style>

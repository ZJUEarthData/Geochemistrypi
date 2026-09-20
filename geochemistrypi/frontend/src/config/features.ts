const configuredAuthUi = import.meta.env.VITE_ENABLE_AUTH_UI

export const authUiEnabled =
  configuredAuthUi === 'true' || (import.meta.env.DEV && configuredAuthUi !== 'false')

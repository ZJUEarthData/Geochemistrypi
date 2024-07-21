import globals from 'globals'
import pluginJs from '@eslint/js'
// import tseslint from "typescript-eslint";
import pluginVue from 'eslint-plugin-vue'

export default [
  { languageOptions: { globals: globals.browser } },
  pluginJs.configs.recommended,
  // ...tseslint.configs.recommended,
  ...pluginVue.configs['flat/essential'],
  {
    rules: {
      semi: ['warn', 'never'],
      'comma-dangle': ['error', 'never'],
      'no-unused-vars': 2,
      'space-before-function-paren': 0,
      'generator-star-spacing': 'off',
      'object-curly-spacing': 0, // Enforce consistent spacing within curly braces
      'array-bracket-spacing': 0 // square brackets
    }
  }
]

import '@mdi/font/css/materialdesignicons.css'
import '@nynxz/zenkit-ui/style.css'
import './app.css'

import { createApp } from 'vue'
import { createRouter, createWebHistory } from 'vue-router'
import App from './App.vue'
import { registerLoraSource } from './loraSource'
import { initSettings } from './settings'
import { initTheme } from './theme'

const router = createRouter({
  history: createWebHistory('/zentrainer/'),
  routes: [
    { path: '/', redirect: '/runs' },
    { path: '/runs', component: () => import('./views/RunsView.vue') },
    { path: '/runs/new', component: () => import('./views/NewRunView.vue') },
    { path: '/queue', redirect: '/runs' },
    { path: '/runs/:id', component: () => import('./views/RunView.vue'), props: true },
    { path: '/datasets', component: () => import('./views/DatasetsView.vue') },
    { path: '/datasets/:name', component: () => import('./views/DatasetView.vue'), props: true },
    { path: '/models', component: () => import('./views/ModelsView.vue') },
    { path: '/gallery', component: () => import('./views/GalleryView.vue') },
    { path: '/apps', component: () => import('./views/AppsView.vue') },
    { path: '/apps/w/txt2img', redirect: '/apps/text-to-image' },
    { path: '/apps/:id', component: () => import('./views/AppView.vue'), props: true },
    { path: '/system', component: () => import('./views/SystemView.vue') },
    { path: '/settings', component: () => import('./views/SettingsView.vue') },
  ],
})

registerLoraSource()
await Promise.all([initTheme(), initSettings()])
createApp(App).use(router).mount('#app')

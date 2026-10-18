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
    { path: '/', redirect: '/jobs' },
    { path: '/jobs', component: () => import('./views/JobsView.vue') },
    { path: '/simple', component: () => import('./views/SimpleView.vue') },
    { path: '/tools', component: () => import('./views/ToolsView.vue') },
    { path: '/tools/merge', component: () => import('./views/MergeView.vue') },
    { path: '/tools/split-videos', component: () => import('./views/SplitVideosView.vue') },
    { path: '/loras/merge', redirect: (to) => ({ path: '/tools/merge', query: to.query }) },
    { path: '/simple/run/:id', component: () => import('./views/SimpleRunView.vue'), props: true },
    { path: '/simple/:id', component: () => import('./views/SimpleView.vue'), props: true },
    { path: '/jobs/caption/:id', component: () => import('./views/CaptionJobView.vue'), props: true },
    { path: '/runs', redirect: '/jobs' },
    { path: '/runs/new', component: () => import('./views/NewRunView.vue') },
    { path: '/queue', redirect: '/jobs' },
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

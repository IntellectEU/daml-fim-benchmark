/**
 * Browser entry point: mounts the React app into the page's #root element and
 * pulls in the global styles (Ant Design's CSS reset plus our own styles.css).
 * This is the file Vite loads first; everything else hangs off <App />.
 */
import React from 'react'
import ReactDOM from 'react-dom/client'
import 'antd/dist/reset.css'
import './styles.css'
import { App } from './App'

ReactDOM.createRoot(document.getElementById('root')!).render(
  <React.StrictMode>
    <App />
  </React.StrictMode>
)

/**
 * Top-level app shell: the page header/title, the nav menu, and the routes.
 * The only page here is the FIM runs page; "/" redirects to "/fim".
 */
import { BrowserRouter, Routes, Route, Navigate, Link } from 'react-router-dom'
import { Layout, Menu } from 'antd'
import { FimPage } from './FimPage'

const { Header, Content } = Layout

export function App() {
  return (
    <BrowserRouter>
      <Layout style={{ minHeight: '100vh' }}>
        <Header style={{ display: 'flex', alignItems: 'center', padding: '0 24px' }}>
          <div style={{ color: 'white', fontWeight: 'bold', fontSize: '18px', marginRight: '32px' }}>
            FIM Benchmark
          </div>
          <Menu
            theme="dark"
            mode="horizontal"
            defaultSelectedKeys={['/fim']}
            items={[
              {
                key: '/fim',
                label: <Link to="/fim">FIM Runs</Link>
              }
            ]}
          />
        </Header>
        <Content style={{ padding: '24px' }}>
          <Routes>
            <Route path="/" element={<Navigate to="/fim" replace />} />
            <Route path="/fim" element={<FimPage />} />
          </Routes>
        </Content>
      </Layout>
    </BrowserRouter>
  )
}

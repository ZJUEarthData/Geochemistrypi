import React, { useEffect, useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import DatasetUploadButton from '../components/Dataset';
// import Command from '../components/Command';
import { MenuFoldOutlined, MenuUnfoldOutlined, DashboardOutlined, UserOutlined, DatabaseOutlined, HomeOutlined } from '@ant-design/icons';
import { Layout, Menu, Button, theme } from 'antd';

const { Header, Sider, Content } = Layout;

const HomePage = () => {
    const [collapsed, setCollapsed] = useState<boolean>(false);
    const {
        token: { colorBgContainer },
    } = theme.useToken();

    useEffect(() => {
        toast.success('Welcome to Geochemistry π! ✨');
    }, []);

    return (
        <Layout>
            <Sider trigger={null} collapsible collapsed={collapsed}>
                <div className="logo" />
                <Menu
                    theme="dark"
                    mode="inline"
                    defaultSelectedKeys={['1']}
                    items={[
                        {
                            key: '1',
                            icon: <HomeOutlined />,
                            label: 'Home',
                        },
                        {
                            key: '2',
                            icon: <DatabaseOutlined />,
                            label: 'Dataset',
                        },
                        {
                            key: '3',
                            icon: <DashboardOutlined />,
                            label: 'Dashbord',
                        },
                        {
                            key: '4',
                            icon: <UserOutlined />,
                            label: 'Setting',
                        },
                    ]}
                    onClick={(item) => {
                        console.log(item);
                    }}
                />
            </Sider>
            <Layout>
                <Header style={{ padding: 0, background: colorBgContainer }}>
                    <Button
                        type="text"
                        icon={collapsed ? <MenuUnfoldOutlined /> : <MenuFoldOutlined />}
                        onClick={() => setCollapsed(!collapsed)}
                        style={{
                            fontSize: '16px',
                            width: 64,
                            height: 64,
                        }}
                    />
                </Header>
                <Content
                    style={{
                        margin: '24px 16px',
                        padding: 24,
                        minHeight: 'calc(100vh - 64px - 48px)',
                        background: colorBgContainer,
                    }}
                >
                    <DatasetUploadButton />
                </Content>
            </Layout>
            <Toaster />
        </Layout>
    );
};

export default HomePage;

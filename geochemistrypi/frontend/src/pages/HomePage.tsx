import React, { useEffect, useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import DatasetUploadButton from '../components/Dataset';
import DatasetDisplay from '../components/DatasetDisplay';
import Dashbord from '../components/Dashboard';
import Profile from '../components/Profile';
// import Command from '../components/Command';
import { MenuFoldOutlined, MenuUnfoldOutlined, DashboardOutlined, UserOutlined, DatabaseOutlined, HomeOutlined } from '@ant-design/icons';
import { Layout, Menu, Button, theme } from 'antd';

const HomePage = () => {
    const [collapsed, setCollapsed] = useState<boolean>(false);
    const [activeMenu, setActiveMenu] = useState<string>('1');
    const [contentComponent, setContentComponent] = useState<React.ReactNode>(<DatasetUploadButton />);
    const { Header, Sider, Content } = Layout;
    const {
        token: { colorBgContainer },
    } = theme.useToken();

    useEffect(() => {
        toast.success('Welcome to Geochemistry π! ✨');
    }, []);

    const handleMenuClick = (item: any) => {
        const { key } = item;
        setActiveMenu(key);

        switch (key) {
            case '1':
                setContentComponent(<DatasetUploadButton />);
                break;
            case '2':
                setContentComponent(<DatasetDisplay />);
                break;
            case '3':
                setContentComponent(<Dashbord />);
                break;
            case '4':
                setContentComponent(<Profile />);
                break;
            default:
                break;
        }
    };

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
                    onClick={handleMenuClick}
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
                    {contentComponent}
                </Content>
            </Layout>
            <Toaster />
        </Layout>
    );
};

export default HomePage;

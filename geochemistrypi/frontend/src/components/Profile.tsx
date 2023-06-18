import React, { useState, useEffect } from 'react';
import { Typography, Button, Row, Col, Divider } from 'antd';
import { getUserInfo } from '../helpers/apiCall';
import { toast } from 'react-hot-toast';
import { useCookies } from 'react-cookie';
import { useNavigate } from 'react-router-dom';

const { Title, Text } = Typography;

const Profile = () => {
    const [userNames, setUserNames] = useState<string>('');
    const [userEmail, setUserEmail] = useState<string>('');
    const [uploadDatasetNum, setUploadDatasetNum] = useState<number>(0);
    const [, , removeCookie] = useCookies(['token']);
    const navigate = useNavigate();

    useEffect(() => {
        const fetchUserInfo = async () => {
            try {
                const response = await getUserInfo();
                if (response.status === 200) {
                    setUserNames(response.data['username']);
                    setUserEmail(response.data['email']);
                    setUploadDatasetNum(response.data['upload_count']);
                } else {
                    toast.error('An error occurred, please try again later');
                }
            } catch (error) {
                toast.error('An error occurred, please try again later');
            }
        };
        fetchUserInfo();
    }, []);

    const handleLogout = () => {
        removeCookie('token');
        navigate('/');
    };

    return (
        <div style={{ display: 'flex', flexDirection: 'column', height: '100vh', padding: '10px' }}>
            <Title level={3}>Profile</Title>
            <Divider />

            <Row gutter={[24, 24]}>
                <Col span={12}>
                    <Text strong>Username:</Text>
                </Col>
                <Col span={12}>
                    <Text>{userNames}</Text>
                </Col>
            </Row>

            <Row gutter={[24, 24]}>
                <Col span={12}>
                    <Text strong>Email:</Text>
                </Col>
                <Col span={12}>
                    <Text>{userEmail}</Text>
                </Col>
            </Row>

            <Row gutter={[24, 24]}>
                <Col span={12}>
                    <Text strong>Number of uploaded datasets:</Text>
                </Col>
                <Col span={12}>
                    <Text>{uploadDatasetNum}</Text>
                </Col>
            </Row>

            <Divider />

            <Row justify="center" align="middle">
                <Col>
                    <Button type="primary" onClick={handleLogout} size="large" block>
                        Logout
                    </Button>
                </Col>
            </Row>
        </div>
    );
};

export default Profile;

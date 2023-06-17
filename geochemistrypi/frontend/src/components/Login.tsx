import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { postLogin } from '../helpers/apiCall';
import { useNavigate } from 'react-router-dom';
import { useCookies } from 'react-cookie';
import { Button, Form, Input } from 'antd';
import { LockOutlined, MailOutlined } from '@ant-design/icons';

const Login = () => {
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const navigate = useNavigate();

    const [cookies, setCookie] = useCookies(['token']);

    const handleSubmit = async () => {
        // validate email and password
        if (email === '' || password === '') {
            toast.error('Email and password cannot be empty!');
            return;
        }
        try {
            const response = await postLogin(email, password);
            if (response.status === 200) {
                setCookie('token', response.data.access_token, { path: '/' });
                // console.log(response.data);
                toast.success('Login successful!');
                navigate('/home');
            } else {
                toast.error('Invalid email or password!');
            }
        } catch (error) {
            toast.error('An error occurred, please try again later');
        }
    };

    const handleEmailChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setPassword(event.target.value);
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <Form name="normal_login" className="login-form" initialValues={{ remember: true }}>
                <Form.Item name="email" rules={[{ required: true, message: 'Please input your Email!' }]}>
                    <Input prefix={<MailOutlined className="site-form-item-icon" />} type="email" placeholder="Email" onChange={handleEmailChange} />
                </Form.Item>
                <Form.Item name="password" rules={[{ required: true, message: 'Please input your Password!' }]}>
                    <Input prefix={<LockOutlined className="site-form-item-icon" />} type="password" placeholder="Password" onChange={handlePasswordChange} />
                </Form.Item>
                <Form.Item>
                    <Button type="primary" htmlType="submit" className="login-form-button" onClick={handleSubmit} style={{ width: 100 }}>
                        Log in
                    </Button>
                </Form.Item>
            </Form>
            <Toaster />
        </div>
    );
};

export default Login;

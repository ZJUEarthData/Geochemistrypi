import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import { postRegister } from '../helpers/apiCall';
import { useCookies } from 'react-cookie';
import { Button, Form, Input } from 'antd';
import { LockOutlined, UserOutlined, MailOutlined } from '@ant-design/icons';

const Register = () => {
    const [name, setName] = useState<string>('');
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [cookies, setCookie] = useCookies(['token']);
    const navigate = useNavigate();

    const handleSubmit = async () => {
        // Validate the form
        if (name === '') {
            toast.error('Please enter your name');
            return;
        }
        if (password === '') {
            toast.error('Please enter your password');
            return;
        }
        if (email === '') {
            toast.error('Please enter your email');
            return;
        }

        try {
            const response = await postRegister(name, email, password);
            if (response.status === 200) {
                setCookie('token', response.data.access_token, { path: '/' });
                toast.success('Registration successful!');
                navigate('/home');
            } else {
                toast.error('Invalid email or password!');
            }
        } catch (error) {
            toast.error('An error occurred, please try again later');
        }
    };

    const handleNameChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setName(event.target.value);
    };

    const handleEmailChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setEmail(event.target.value);
    };

    const handlePasswordChange = (event: React.ChangeEvent<HTMLInputElement>) => {
        setPassword(event.target.value);
    };

    return (
        <div style={{ display: 'flex', justifyContent: 'center', alignItems: 'center', height: '100vh' }}>
            <Form name="normal_register" className="register-form" initialValues={{ remember: true }}>
                <Form.Item name="name" rules={[{ required: true, message: 'Please input your Name!' }]}>
                    <Input prefix={<UserOutlined className="site-form-item-icon" />} type="name" placeholder="Name" onChange={handleNameChange} />
                </Form.Item>
                <Form.Item name="email" rules={[{ required: true, message: 'Please input your Email!' }]}>
                    <Input prefix={<MailOutlined className="site-form-item-icon" />} type="email" placeholder="Email" onChange={handleEmailChange} />
                </Form.Item>
                <Form.Item name="password" rules={[{ required: true, message: 'Please input your Password!' }]}>
                    <Input prefix={<LockOutlined className="site-form-item-icon" />} type="password" placeholder="Password" onChange={handlePasswordChange} />
                </Form.Item>
                <Form.Item>
                    <Button type="primary" htmlType="submit" className="register-form-button" onClick={handleSubmit} style={{ width: 100 }}>
                        Register
                    </Button>
                </Form.Item>
            </Form>
            <Toaster />
        </div>
    );
};

export default Register;

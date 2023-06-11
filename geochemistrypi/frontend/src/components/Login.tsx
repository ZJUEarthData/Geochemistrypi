import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { postLogin } from '../helpers/apiCall';
import { useNavigate } from 'react-router-dom';
// import { setUserIDCookie } from "../helpers/cookies";
import { useCookies } from 'react-cookie';

const Login = () => {
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const navigate = useNavigate();

    const [cookies, setCookie] = useCookies(['userID']);

    const handleSubmit = async () => {
        // validate email and password
        if (email === '' || password === '') {
            toast.error('Email and password cannot be empty!');
            return;
        }
        try {
            const response = await postLogin(email, password);
            if (response.status === 200) {
                // setUserIDCookie(response.data.userID);
                setCookie('userID', response.data.userID, { path: '/' });
                console.log(response.data.userID);
                console.log(response.data);
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
        <div>
            <h1>Login</h1>
            <input type="email" id="emailInput" placeholder="Email" onChange={handleEmailChange} />
            <input type="password" id="passwordInput" placeholder="Password" onChange={handlePasswordChange} />
            <button onClick={handleSubmit}>Login</button>
            <Toaster />
        </div>
    );
};

export default Login;

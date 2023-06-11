import React, { useState } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import { useNavigate } from 'react-router-dom';
import { postRegister } from '../helpers/apiCall';
// import { setUserIDCookie } from "../helpers/cookies";
import { useCookies } from 'react-cookie';

const Register = () => {
    const [name, setName] = useState<string>('');
    const [email, setEmail] = useState<string>('');
    const [password, setPassword] = useState<string>('');
    const [cookies, setCookie] = useCookies(['userID']);
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
                // setUserIDCookie(response.data.userID);
                setCookie('userID', response.data.userID, { path: '/' });
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
        <div>
            <h1>Register</h1>
            <input type="text" id="nameInput" placeholder="Name" onChange={handleNameChange} />
            <input type="email" id="emailInput" placeholder="Email" onChange={handleEmailChange} />
            <input type="password" id="passwordInput" placeholder="Password" onChange={handlePasswordChange} />
            <button onClick={handleSubmit}>Register</button>
            <Toaster />
        </div>
    );
};

export default Register;

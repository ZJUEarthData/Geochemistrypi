import React, { useState } from 'react';
import Login from '../components/Login';
import Register from '../components/Register';
import { HeaderButton } from '../components/Header';
import { PAGE_STATUS } from '../helpers/constants';

const LoginPage = () => {
    const [showLogin, setShowLogin] = useState<boolean>(true);

    const handleRegister = () => {
        setShowLogin(false);
    };

    const handleLogin = () => {
        setShowLogin(true);
    };

    return (
        <div>
            <HeaderButton page_status={PAGE_STATUS.login} handleRegister={handleRegister} handleLogin={handleLogin} />
            {showLogin ? <Login /> : <Register />}
        </div>
    );
};

export default LoginPage;

import React from 'react';
import { PAGE_STATUS } from '../helpers/constants';

type onClickFunction = (event: React.MouseEvent<HTMLButtonElement>) => void;

interface HeaderButtonProps {
    page_status: number;
    handleRegister?: onClickFunction;
    handleLogin?: onClickFunction;
}

export const HeaderButton = (props: HeaderButtonProps) => {
    const { page_status, handleRegister, handleLogin } = props;

    return (
        <div>
            {/* {page_status === PAGE_STATUS.login ? <button onClick={handleRegister}>Register</button> : <button onClick={handleLogin}>Login</button>} */}
            {page_status === PAGE_STATUS.login && <button onClick={handleRegister}>Register</button>}
            {<button onClick={handleLogin}>{page_status === PAGE_STATUS.login ? 'Login' : 'Logout'}</button>}
        </div>
    );
};

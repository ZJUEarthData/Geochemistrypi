import React, { useEffect } from 'react';
import { useLocation } from 'react-router-dom';

const RefreshHandler = () => {
    const location = useLocation();

    useEffect(() => {
        const handleBeforeUnload = (event: BeforeUnloadEvent) => {
            event.preventDefault();
            event.returnValue = ''; // Chrome requires returnValue to be set
        };

        const handleRefresh = (event) => {
            event.preventDefault();
            const confirmationMessage = 'Are you sure you want to refresh?'; // Confirmation message

            event.returnValue = confirmationMessage;
            return confirmationMessage;
        };

        window.addEventListener('beforeunload', handleBeforeUnload);
        window.addEventListener('unload', handleRefresh);

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            window.removeEventListener('unload', handleRefresh);
        };
    }, []);

    useEffect(() => {
        if (location.pathname === '/home') {
            const currentState = window.history.state;
            window.history.replaceState(currentState, '');
        }
    }, [location]);

    return null;
};

export default RefreshHandler;

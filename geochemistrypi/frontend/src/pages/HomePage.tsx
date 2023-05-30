import React, { useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import DataUploadButton from '../components/Data';

const HomePage = () => {
    useEffect(() => {
        toast.success('Welcome to Geochemistry π! ✨');
    }, []);

    return (
        <div>
            <DataUploadButton />
            <Toaster />
        </div>
    );
};

export default HomePage;

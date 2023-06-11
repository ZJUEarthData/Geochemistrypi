import React, { useEffect } from 'react';
import toast, { Toaster } from 'react-hot-toast';
import DatasetUploadButton from '../components/Dataset';
import Command from '../components/Command';

const HomePage = () => {
    useEffect(() => {
        toast.success('Welcome to Geochemistry π! ✨');
    }, []);

    return (
        <div>
            <DatasetUploadButton />
            {/* <Command /> */}
            <Toaster />
        </div>
    );
};

export default HomePage;

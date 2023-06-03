import React, { ChangeEvent, useState, useEffect } from 'react';
import axios from 'axios';
import DataFrame from './DataFrame';

const DataUploadButton = () => {
    const [file, setFile] = useState<File | null>(null);
    const [show, setShow] = useState<boolean>(false);
    const [processedData, setProcessedData] = useState<any[]>([]);

    // Get the hostname of the current page
    const hostname = window.location.hostname;
    const backendPort = '8000';
    let baseURL = window.location.protocol + '//' + hostname + ':' + backendPort;
    const dashURL = baseURL + '/dash/';

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = event.target.files?.[0];
        setFile(selectedFile || null);
    };

    const postFile = async () => {
        if (file) {
            const formData = new FormData();
            formData.append('data', new Blob([file], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
            try {
                const response = await axios.post('http://0.0.0.0:8000/data-mining/upload', formData, {
                    headers: {
                        'Content-Type': 'multipart/form-data',
                    },
                });
                // console.log(typeof JSON.parse(response.data));
                setProcessedData(JSON.parse(response.data));
            } catch (error) {
                console.log(error);
            }
        } else {
            console.log('No file selected');
        }
    };

    const handleFileUpload = () => {
        // console.log(file);
        if (file) {
            postFile();
            setShow(true);
        }
    };

    const containerStyle: React.CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
    };

    const buttonStyle: React.CSSProperties = {
        // width: '100px',
        // height: '50px',
    };

    return (
        <div style={containerStyle}>
            <h2>Geochemistry Pi - Data Uploading</h2>
            <br />
            <input type="file" accept=".xlsx" onChange={handleFileChange} />
            <br />
            <button type="submit" disabled={!file} onClick={handleFileUpload} style={buttonStyle}>
                Upload
            </button>
            {show && <hr />}
            {show && <p>File uploaded!</p>}
            {/* {processedData.length > 0 && <DataFrame data={processedData} />} */}
            {show && <button onClick={() => (window.location.href = dashURL)}>Visualize</button>}
        </div>
    );
};

export default DataUploadButton;

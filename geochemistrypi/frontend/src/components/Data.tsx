import React, { ChangeEvent, useState, useEffect } from 'react';
import axios from 'axios';
import DataFrame from './DataFrame';

const DataUploadButton = () => {
    const [file, setFile] = useState<File | null>(null);
    const [show, setShow] = useState<boolean>(false);
    const [processedData, setProcessedData] = useState<any[]>([]);

    const handleFileChange = (event: ChangeEvent<HTMLInputElement>) => {
        const selectedFile = event.target.files?.[0];
        setFile(selectedFile || null);
    };

    const postFile = async () => {
        if (file) {
            const formData = new FormData();
            formData.append('data', new Blob([file], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
            try {
                const response = await axios.post('http://0.0.0.0:8000/data/upload', formData, {
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

    return (
        <div>
            <h2>Upload Data</h2>
            <input type="file" accept=".xlsx" onChange={handleFileChange} />
            <button type="submit" disabled={!file} onClick={handleFileUpload}>
                Upload
            </button>
            {show && <hr />}
            {show && <p>File uploaded!</p>}
            {processedData.length > 0 && <DataFrame data={processedData} />}
        </div>
    );
};

export default DataUploadButton;

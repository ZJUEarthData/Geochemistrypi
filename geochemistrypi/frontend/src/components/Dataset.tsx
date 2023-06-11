import React, { ChangeEvent, useState, useEffect } from 'react';
// import axios from 'axios';
// import DataFrame from './DataFrame';
import { postDataset } from '../helpers/apiCall';
// import { Cookies } from 'react-cookie';
import { useCookies } from 'react-cookie';
import { toast } from 'react-hot-toast';

const DatasetUploadButton = () => {
    const [dataset, setDataset] = useState<File | null>(null);
    const [show, setShow] = useState<boolean>(false);
    const [processedData, setProcessedData] = useState<any[]>([]);
    const [cookies, setCookie] = useCookies(['userID']);

    // Get the hostname of the current page
    const hostname = window.location.hostname;
    const backendPort = '8000';
    let baseURL = window.location.protocol + '//' + hostname + ':' + backendPort;
    const dashURL = baseURL + '/dash/';

    const handleDatasetChange = (event: ChangeEvent<HTMLInputElement>) => {
        const selectedDataset = event.target.files?.[0];
        setDataset(selectedDataset || null);
    };

    // const postFile = async () => {
    //     if (dataset) {
    //         const formData = new FormData();
    //         formData.append('data', new Blob([dataset], { type: 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet' }));
    //         try {
    //             const response = await axios.post('http://0.0.0.0:8000/data-mining/upload', formData, {
    //                 headers: {
    //                     'Content-Type': 'multipart/form-data',
    //                 },
    //             });
    //             // console.log(typeof JSON.parse(response.data));
    //             setProcessedData(JSON.parse(response.data));
    //         } catch (error) {
    //             console.log(error);
    //         }
    //     } else {
    //         console.log('No file selected');
    //     }
    // };

    const handleDatasetUpload = async () => {
        // console.log(file);
        if (dataset) {
            let userID = cookies['userID'];
            // console.log('Dataset: ' + userID);
            try {
                const response = await postDataset(dataset, userID);
                // console.log(response);
                if (response.status === 429) {
                    toast.error('You have exceeded the maximum number of datasets allowed. Please delete a dataset before uploading a new one.');
                }
            } catch (error) {
                console.log(error);
            }
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
            <input type="file" accept=".xlsx" onChange={handleDatasetChange} />
            <br />
            <button type="submit" disabled={!dataset} onClick={handleDatasetUpload} style={buttonStyle}>
                Upload
            </button>
            {show && <hr />}
            {show && <p>File uploaded!</p>}
            {/* {processedData.length > 0 && <DataFrame data={processedData} />} */}
            {show && <button onClick={() => (window.location.href = dashURL)}>Visualize</button>}
        </div>
    );
};

export default DatasetUploadButton;

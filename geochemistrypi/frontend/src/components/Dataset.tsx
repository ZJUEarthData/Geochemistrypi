import React, { ChangeEvent, useState, useEffect } from 'react';
// import axios from 'axios';
// import DataFrame from './DataFrame';
import { postDataset } from '../helpers/apiCall';
// import { Cookies } from 'react-cookie';
import { useCookies } from 'react-cookie';
import { toast } from 'react-hot-toast';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { message, Upload } from 'antd';

const DatasetUploadButton = () => {
    const [dataset, setDataset] = useState<File | null>(null);
    const [show, setShow] = useState<boolean>(false);
    const [processedData, setProcessedData] = useState<any[]>([]);
    const [cookies, setCookie] = useCookies(['userID']);
    const { Dragger } = Upload;
    const userID = cookies['userID'];

    // Get the hostname of the current page
    const hostname = window.location.hostname;
    const backendPort = '8000';
    let baseURL = window.location.protocol + '//' + hostname + ':' + backendPort;
    const dashURL = baseURL + '/dash/';

    const handleDatasetChange = (event: ChangeEvent<HTMLInputElement>) => {
        const selectedDataset = event.target.files?.[0];
        setDataset(selectedDataset || null);
    };

    // const handleDatasetUpload = async () => {
    //     // console.log(file);
    //     if (dataset) {
    //         let userID = cookies['userID'];
    //         // console.log('Dataset: ' + userID);
    //         try {
    //             const response = await postDataset(dataset, userID);
    //             // console.log(response);
    //             if (response.status === 429) {
    //                 toast.error('You have exceeded the maximum number of datasets allowed. Please delete a dataset before uploading a new one.');
    //             }
    //         } catch (error) {
    //             console.log(error);
    //         }
    //         setShow(true);
    //     }
    // };

    const containerStyle: React.CSSProperties = {
        display: 'flex',
        flexDirection: 'column',
        justifyContent: 'center',
        alignItems: 'center',
        height: '100vh',
    };

    const props: UploadProps = {
        name: 'file',
        multiple: false,
        customRequest: async (options) => {
            const { onSuccess, onError, file } = options;
            try {
                const response = await postDataset(file, userID);
                if (response.status === 429) {
                    message.error(`${file.name} file upload failed`);
                    message.error('You have exceeded the maximum number of datasets allowed. Please delete a dataset before uploading a new one.');
                } else {
                    message.success('File uploaded successfully.');
                }
                onSuccess('Ok');
            } catch (error) {
                console.log(error);
                onError('Error');
            }
        },
        onDrop(e) {
            console.log('Dropped files', e.dataTransfer.files);
        },
    };

    return (
        <div style={containerStyle}>
            <Dragger {...props}>
                <p className="ant-upload-drag-icon">
                    <InboxOutlined />
                </p>
                <p className="ant-upload-text">Click or drag file to this area to upload</p>
                <p className="ant-upload-hint">Support for a single file (E.g. dataset.xlsx) upload.</p>
            </Dragger>
            {/* <h2>Geochemistry Pi - Data Uploading</h2>
            <br />
            <input type="file" accept=".xlsx" onChange={handleDatasetChange} />
            <br />
            <button type="submit" disabled={!dataset} onClick={handleDatasetUpload} style={buttonStyle}>
                Upload
            </button>
            {show && <hr />}
            {show && <p>File uploaded!</p>}
            {processedData.length > 0 && <DataFrame data={processedData} />}
            {show && <button onClick={() => (window.location.href = dashURL)}>Visualize</button>} */}
        </div>
    );
};

export default DatasetUploadButton;

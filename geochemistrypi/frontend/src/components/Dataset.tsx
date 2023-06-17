import React from 'react';
import { postDataset } from '../helpers/apiCall';
import { InboxOutlined } from '@ant-design/icons';
import type { UploadProps } from 'antd';
import { message, Upload } from 'antd';

const DatasetUploadButton = () => {
    const { Dragger } = Upload;

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
                const response = await postDataset(file);
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
        </div>
    );
};

export default DatasetUploadButton;

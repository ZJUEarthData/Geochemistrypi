import { Row, Col, Card } from 'antd';
import React from 'react';

const DatasetDisplay = () => {
    const datasetFiles = [
        { id: 1, name: 'dataset1.xlsx' },
        { id: 2, name: 'dataset2.xlsx' },
        { id: 3, name: 'dataset3.xlsx' },
        { id: 4, name: 'dataset4.xlsx' },
        { id: 5, name: 'dataset5.xlsx' },
    ];

    const handleClickDataset = (file) => {
        // Perform the necessary action when a dataset file is clicked
        // For example, navigate to a different page or show the data in a modal
        console.log('Clicked dataset file:', file);
    };

    return (
        <div>
            <Row gutter={[16, 16]}>
                {datasetFiles.map((file) => (
                    <Col key={file.id} xs={24} sm={12} md={8} lg={6} xl={4}>
                        <Card title={file.name} hoverable onClick={() => handleClickDataset(file)}>
                            {/* Additional content or thumbnail can be added here */}
                        </Card>
                    </Col>
                ))}
            </Row>
        </div>
    );
};

export default DatasetDisplay;

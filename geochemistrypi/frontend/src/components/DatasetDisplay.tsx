import { Row, Col, Card, Button, Modal, Table } from 'antd';
import React, { useState, useEffect } from 'react';
import { getBasicDatasetsInfo, deleteDataset, getDataset } from '../helpers/apiCall';
import { toast } from 'react-hot-toast';

const DatasetDisplay = () => {
    const [datasetInfo, setDatasetInfo] = useState<any[]>([]);
    const [selectedDataset, setSelectedDataset] = useState<any[]>([]);
    const [selectedDatasetName, setSelectedDatasetName] = useState<string>('');
    const [isModalVisible, setIsModalVisible] = useState<boolean>(false);

    useEffect(() => {
        const fetchDatasetInfo = async () => {
            try {
                const response = await getBasicDatasetsInfo();
                if (response.status === 200) {
                    setDatasetInfo(response.data);
                } else {
                    toast.error('An error occurred, please try again later');
                }
            } catch (error) {
                toast.error('An error occurred, please try again later');
            }
        };
        fetchDatasetInfo();
    }, []);

    const handleClickViewDataset = async (datasetID: number) => {
        try {
            const response = await getDataset(datasetID);
            // console.log(response);
            if (response.status === 200) {
                toast.success('Dataset loaded successfully!');
                setSelectedDataset(JSON.parse(response.data.dataset['json_data']));
                setSelectedDatasetName(response.data.dataset['name']);
                setIsModalVisible(true);
                // console.log(selectedDataset);
            } else {
                toast.error('An error occurred, please try again later');
            }
        } catch (error) {
            toast.error('An error occurred, please try again later');
        }
    };

    const handleModalClose = () => {
        setIsModalVisible(false);
    };

    const handleClickDeleteDataset = async (datasetID: number) => {
        try {
            const response = await deleteDataset(datasetID);
            if (response.status === 200) {
                toast.success('Dataset deleted successfully!');
                const newDatasetInfo = datasetInfo.filter((dataset) => dataset.id !== datasetID);
                setDatasetInfo(newDatasetInfo);
            } else {
                toast.error('An error occurred, please try again later');
            }
        } catch (error) {
            toast.error('An error occurred, please try again later');
        }
    };

    return (
        <div>
            <Row gutter={[16, 16]}>
                {datasetInfo.map((dataset) => (
                    <Col key={dataset.sequence} xs={24} sm={12} md={8} lg={6} xl={4}>
                        <Card
                            title={dataset.name}
                            hoverable
                            actions={[
                                <Button type="link" onClick={() => handleClickViewDataset(dataset.id)}>
                                    View
                                </Button>,
                                <Button type="link" danger onClick={() => handleClickDeleteDataset(dataset.id)}>
                                    Delete
                                </Button>,
                            ]}
                        >
                            {/* Additional content or thumbnail can be added here */}
                        </Card>
                    </Col>
                ))}
            </Row>
            <Modal title={selectedDatasetName ? selectedDatasetName : ''} open={isModalVisible} onOk={handleModalClose} onCancel={handleModalClose} footer={null}>
                <div style={{ overflowX: 'scroll', overflowY: 'scroll' }}>
                    <p>Showing the first 5 rows:</p>
                    {selectedDataset && (
                        <Table
                            dataSource={selectedDataset.slice(0, 5)}
                            columns={selectedDataset[0] ? Object.keys(selectedDataset[0]).map((key, index) => ({ title: key, dataIndex: key, key: `column-${index}` })) : []}
                            pagination={false}
                        />
                    )}
                </div>
            </Modal>
        </div>
    );
};

export default DatasetDisplay;

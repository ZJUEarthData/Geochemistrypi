import { Row, Col, Card, Button } from 'antd';
import React, { useState, useEffect } from 'react';
import { getBasicDatasetsInfo, deleteDataset } from '../helpers/apiCall';
import { useCookies } from 'react-cookie';
import { toast } from 'react-hot-toast';

const DatasetDisplay = () => {
    const [datasetInfo, setDatasetInfo] = useState<any[]>([]);
    const [cookies, setCookie] = useCookies(['userID']);
    const userID = cookies['userID'];

    useEffect(() => {
        const fetchDatasetInfo = async () => {
            try {
                const response = await getBasicDatasetsInfo(userID);
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

    const handleClickDeleteDataset = async (datasetID: number) => {
        try {
            const response = await deleteDataset(userID, datasetID);
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
        </div>
    );
};

export default DatasetDisplay;

import React from 'react';

interface DataFrameProps {
    data: any[];
}

const DataFrameDisplay = ({ data }: DataFrameProps) => {
    const renderTableHeader = () => {
        if (!data || !Array.isArray(data) || data.length === 0) {
            // console.log(Array.isArray(data))
            return <p>No data available</p>;
        }

        const columns = Object.keys(data[0]);

        return (
            <table className="data-frame">
                <thead>
                    <tr>
                        {columns.map((column) => (
                            <th key={column}>{column}</th>
                        ))}
                    </tr>
                </thead>
                <tbody>
                    {data.map((row, index) => (
                        <tr key={index}>
                            {columns.map((column) => (
                                <td key={column}>{row[column]}</td>
                            ))}
                        </tr>
                    ))}
                </tbody>
            </table>
        );
    };

    return (
        <div className="data-frame-container">
            <h2>Processed Data</h2>
            {renderTableHeader()}
        </div>
    );
};

export default DataFrameDisplay;

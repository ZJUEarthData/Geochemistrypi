import React from 'react';

const Dashbord = () => {
    const dashboardUrl = process.env.REACT_APP_DASHBOARD_URL || 'http://localhost:8000/dash';

    return <iframe src={dashboardUrl} title="Dashboard" width="100%" height="100%" frameBorder="0"></iframe>;
};

export default Dashbord;

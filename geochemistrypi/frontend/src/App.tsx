import React from 'react';
import { HashRouter as Router, Routes, Route } from 'react-router-dom';
import HomePage from './pages/HomePage';
import LoginPage from './pages/LoginPage';
import RefreshHandler from './components/RefreshHandler';

const App = () => {
    return (
        <div className="App">
            <Router>
                <RefreshHandler />
                <Routes>
                    <Route path="/" element={<LoginPage />} />
                    <Route path="/home" element={<HomePage />} />
                </Routes>
            </Router>
        </div>
    );
};

export default App;

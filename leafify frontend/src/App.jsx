import React from 'react';
import { BrowserRouter, Routes, Route } from 'react-router-dom';
import Landing from './pages/Landing';
import Scanner from './pages/Scanner';
import AnalysisResult from './pages/AnalysisResult';

export default function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<Landing />} />
        <Route path="/scanner" element={<Scanner />} />
        <Route path="/result" element={<AnalysisResult />} />
      </Routes>
    </BrowserRouter>
  );
}
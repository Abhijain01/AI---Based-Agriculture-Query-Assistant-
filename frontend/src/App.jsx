
import React, { useState, useEffect } from "react";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";

import LoginPage from "./components/LoginPae";
import OtpVerification from "./components/OtpVerificaton";
import RegisterPage from "./components/registerPage";
import RegisterDetails from "./components/RegisterDetails";
import Dashboard from "./components/Dashboard";
import Loader from "./components/Loader"; // 👈 Make sure this path is correct

import "./styles/global.css";

function App() {
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    console.log("⏳ Showing loader...");
    const timer = setTimeout(() => {
      console.log("✅ Hiding loader");
      setLoading(false);
    }, 4000); // loader visible for 4s
    return () => clearTimeout(timer);
  }, []);

  if (loading) {
    return <Loader />; // 👈 should show your video loader here
  }

  return (
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<LoginPage />} />
        <Route path="/otp" element={<OtpVerification />} />
        <Route path="/register" element={<RegisterPage />} />
        <Route path="/register-details" element={<RegisterDetails />} />
        <Route path="/dashboard" element={<Dashboard />} />
        <Route path="/app/*" element={<Dashboard />} />
        <Route path="*" element={<Navigate to="/" replace />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;

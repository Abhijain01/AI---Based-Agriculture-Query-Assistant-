import React, { useState } from "react";
import "../styles/LoginPage.css";
import backgroundImage from "../assets/farm.png";
import logo from "../assets/log_nobg.jpg";
import { FaPhoneVolume } from "react-icons/fa6";
import { useNavigate } from "react-router-dom";

const LoginPage = () => {
  const [phone, setPhone] = useState("");
  const [acceptedTerms, setAcceptedTerms] = useState(false);
  const [loading, setLoading] = useState(false); // For loading state
  const [error, setError] = useState(""); // For displaying errors in the UI
  const navigate = useNavigate();

  // Use an environment variable for the API URL
  const API_URL = process.env.REACT_APP_API_URL || "https://ai-based-agriculture-query-assistant.onrender.com";

  const handleSendOtp = async () => {
    setError(""); // Clear previous errors

    // --- Validation Checks ---
    if (!phone) {
      setError("Please enter your phone number!");
      return;
    }
    const phoneRegex = /^[6-9]\d{9}$/;
    if (!phoneRegex.test(phone)) {
      setError("Please enter a valid 10-digit mobile number!");
      return;
    }
    if (!acceptedTerms) {
      setError("Please accept the terms & conditions to proceed!");
      return;
    }

    setLoading(true);
    try {
      const res = await fetch(`${API_URL}/send-otp`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ phone }),
      });

      const data = await res.json();
      if (res.ok) {
        console.log("✅ OTP sent:", data.otp); // Still useful for debugging
        localStorage.setItem("phone", phone);
        navigate("/otp", { state: { phone } });
      } else {
        setError(data.error || "Failed to send OTP. Please try again.");
      }
    } catch (err) {
      console.error("Fetch Error:", err);
      setError("Something went wrong. Check your connection or try again later.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="login-page">
      <div
        className="image-box"
        style={{ backgroundImage: `url(${backgroundImage})` }}
      >
        <div className="logo">
          <img src={logo} alt="App Logo" />
        </div>
        <div className="login-card">
          <h2>Login</h2>
          <div className="input-group">
            <h4>Phone Number</h4>
            <input
              type="tel"
              placeholder="Enter your phone number"
              value={phone}
              onChange={(e) => setPhone(e.target.value)}
              disabled={loading}
            />
            <span className="phone-icon">
              <FaPhoneVolume />
            </span>
          </div>

          <div className="checkbox">
            <input
              type="checkbox"
              id="terms"
              checked={acceptedTerms}
              onChange={(e) => setAcceptedTerms(e.target.checked)}
              disabled={loading}
            />
            <label htmlFor="terms">
              I accept <a href="/terms">terms & conditions</a>
            </label>
          </div>
          
          {/* Display error message here */}
          {error && <p className="error-message">{error}</p>}

          <button
            className="send-otp"
            onClick={handleSendOtp}
            disabled={loading}
          >
            {loading ? "SENDING..." : "SEND OTP"}
          </button>
        </div>
      </div>
    </div>
  );
};

export default LoginPage;

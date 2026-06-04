import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { BrowserRouter as Router, Routes, Route, Link, useLocation } from 'react-router-dom';
import { 
  AreaChart, Area, BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell 
} from 'recharts';
import './App.css';

// --- NEW: SHORT BEEP SOUND (Base64) ---
// This avoids needing an external MP3 file
const ALERT_SOUND_URL = "data:audio/wav;base64,UklGRl9vT19XQVZFZm10IBAAAAABAAEAQB8AAEAfAAABAAgAZGF0YU"; 

// --- COMPONENT: DASHBOARD (Home) ---
const Dashboard = ({ isDark, isSoundOn }) => {
  const [stats, setStats] = useState({ total_packets: 0, malicious_detected: 0, active_threats: 0, system_health: "100%" });
  const [trafficData, setTrafficData] = useState([]);
  const [recentAlerts, setRecentAlerts] = useState([]);
  
  // Create audio reference
  const audioRef = useRef(new Audio(ALERT_SOUND_URL));

  useEffect(() => {
    const fetchData = async () => {
      try {
        const statRes = await axios.get('http://localhost:8000/api/stats');
        setStats(statRes.data);

        const packetRes = await axios.get('http://localhost:8000/api/live-traffic');
        const packet = packetRes.data;

        setTrafficData(prev => [...prev, { time: packet.timestamp, duration: packet.flow_duration }].slice(-20));

        // --- CHECK FOR ATTACK ---
        if (packet.prediction && packet.prediction !== "BENIGN") {
          
          // 1. ADD TO LIST
          const newAlert = { id: Date.now(), type: packet.prediction, confidence: packet.confidence, timestamp: packet.timestamp };
          setRecentAlerts(prev => [newAlert, ...prev].slice(0, 5));

          // 2. PLAY SOUND (If enabled)
          if (isSoundOn) {
            try {
              // Reset time to 0 to allow rapid re-playing
              audioRef.current.currentTime = 0;
              // We use a beep URL (you can replace the Base64 string above with a real .mp3 url if you want)
              // For now, let's use a standard browser beep approach or just log it if the string is empty
              // A simple trick for a beep without files:
              const context = new (window.AudioContext || window.webkitAudioContext)();
              const oscillator = context.createOscillator();
              const gainNode = context.createGain();
              
              oscillator.connect(gainNode);
              gainNode.connect(context.destination);
              
              oscillator.type = "sine";
              oscillator.frequency.value = 800; // Pitch
              gainNode.gain.value = 0.1; // Volume
              
              oscillator.start();
              setTimeout(() => oscillator.stop(), 150); // Beep for 150ms
              
            } catch (err) {
              console.error("Audio blocked by browser (User must interact with page first)");
            }
          }
        }
      } catch (e) { console.log("Backend offline"); }
    };
    const interval = setInterval(fetchData, 1000);
    return () => clearInterval(interval);
  }, [isSoundOn]); // Re-run logic if sound setting changes

  return (
    <div className="fade-in">
      <header>
        <h1>Network Traffic Overview</h1>
        <div className="user">Admin</div>
      </header>
      
      <div className="kpi-grid">
        <div className="card"><h3>Total Packets</h3><div className="val">{stats.total_packets}</div></div>
        <div className="card danger"><h3>Malicious Detected</h3><div className="val">{stats.malicious_detected}</div></div>
        <div className="card warning"><h3>Active Threats</h3><div className="val">{stats.active_threats}</div></div>
        <div className="card"><h3>System Health</h3><div className="val" style={{color: stats.system_health==="100%"?'#4caf50':'#f44336'}}>{stats.system_health}</div></div>
      </div>

      <div className="charts-section">
        <div className="panel chart-panel">
          <h3>Live Flow Duration</h3>
          <ResponsiveContainer width="100%" height={250}>
            <AreaChart data={trafficData}>
              <defs>
                <linearGradient id="colorFlow" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="5%" stopColor="#6c5ce7" stopOpacity={0.8}/>
                  <stop offset="95%" stopColor="#6c5ce7" stopOpacity={0}/>
                </linearGradient>
              </defs>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={isDark ? "#444" : "#eee"} />
              <XAxis dataKey="time" stroke={isDark ? "#aaa" : "#666"} />
              <YAxis stroke={isDark ? "#aaa" : "#666"} />
              <Tooltip contentStyle={{backgroundColor: isDark ? '#333' : '#fff', border: 'none'}} />
              <Area type="monotone" dataKey="duration" stroke="#6c5ce7" fill="url(#colorFlow)" />
            </AreaChart>
          </ResponsiveContainer>
        </div>

        <div className="panel alerts-panel">
          <h3>Recent Intrusions</h3>
          {recentAlerts.length === 0 ? <p className="no-data">System Secure</p> : (
            <div className="alert-list">
              {recentAlerts.map(a => (
                <div key={a.id} className="alert-item">
                  <div className="icon">🚨</div>
                  <div className="info"><strong>{a.type}</strong><span>{a.timestamp} | {a.confidence}%</span></div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

// --- COMPONENT: ATTACK HISTORY ---
// --- COMPONENT: ATTACK HISTORY (Updated with Download) ---
const History = () => {
  const [history, setHistory] = useState([]);

  useEffect(() => {
    axios.get('http://localhost:8000/api/attack-history')
      .then(res => setHistory(res.data))
      .catch(e => console.log(e));
  }, []);

  // NEW: Function to generate and download CSV
  const downloadCSV = () => {
    if (history.length === 0) {
      alert("No data to download!");
      return;
    }

    // 1. Define Headers
    const headers = ["ID,Timestamp,Attack Type,Confidence,Details,Status"];

    // 2. Convert Data to CSV Format
    const rows = history.map(row => 
      `${row.id},${row.timestamp},${row.type},${row.confidence}%,${row.details.replace(/,/g, '')},Blocked`
    );

    // 3. Join with newlines
    const csvContent = [headers, ...rows].join("\n");

    // 4. Create Blob and Link
    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = window.URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `IDS_Attack_Log_${new Date().toISOString().slice(0,10)}.csv`;
    a.click();
    window.URL.revokeObjectURL(url);
  };

  return (
    <div className="fade-in">
      <header style={{display: 'flex', justifyContent: 'space-between', alignItems: 'center'}}>
        <h1>Attack History Log</h1>
        {/* NEW: Download Button */}
        <button className="download-btn" onClick={downloadCSV}>
          📥 Download CSV
        </button>
      </header>

      <div className="panel full-width">
        <table className="history-table">
          <thead>
            <tr>
              <th>ID</th><th>Time</th><th>Attack Type</th><th>Confidence</th><th>Details</th><th>Status</th>
            </tr>
          </thead>
          <tbody>
            {history.length === 0 ? <tr><td colSpan="6" style={{textAlign:'center'}}>No attacks recorded yet.</td></tr> : 
             history.map((row) => (
              <tr key={row.id}>
                <td>#{row.id}</td><td>{row.timestamp}</td>
                <td style={{fontWeight:'bold', color: '#ff6b6b'}}>{row.type}</td>
                <td>{row.confidence}%</td><td>{row.details}</td>
                <td><span className="badge">Blocked</span></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
};
// --- COMPONENT: MODEL PERFORMANCE ---
const Performance = ({ isDark }) => {
  const metrics = [{ name: 'Accuracy', val: 99.2 }, { name: 'Precision', val: 98.5 }, { name: 'Recall', val: 98.8 }, { name: 'F1-Score', val: 98.6 }];
  const confusionData = [{ name: 'Benign', value: 8500 }, { name: 'DoS', value: 1200 }, { name: 'PortScan', value: 300 }, { name: 'Botnet', value: 50 }];

  return (
    <div className="fade-in">
      <header><h1>Model Performance Analytics</h1></header>
      <div className="perf-grid">
        <div className="panel">
          <h3>Overall Metrics</h3>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={metrics}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} stroke={isDark ? "#444" : "#eee"}/>
              <XAxis dataKey="name" stroke={isDark ? "#aaa" : "#666"} />
              <YAxis domain={[90, 100]} stroke={isDark ? "#aaa" : "#666"} />
              <Tooltip cursor={{fill: 'transparent'}} contentStyle={{backgroundColor: isDark ? '#333' : '#fff'}}/>
              <Bar dataKey="val" fill="#00d2d3" radius={[5, 5, 0, 0]} barSize={50} />
            </BarChart>
          </ResponsiveContainer>
        </div>
        <div className="panel">
          <h3>Class Distribution</h3>
          <ResponsiveContainer width="100%" height={300}>
            <PieChart>
              <Pie data={confusionData} cx="50%" cy="50%" outerRadius={100} fill="#8884d8" dataKey="value" label>
                {confusionData.map((entry, index) => (
                  <Cell key={`cell-${index}`} fill={['#0984e3', '#d63031', '#fdcb6e', '#6c5ce7'][index % 4]} />
                ))}
              </Pie>
              <Tooltip />
            </PieChart>
          </ResponsiveContainer>
        </div>
      </div>
    </div>
  );
};

// --- COMPONENT: SETTINGS (FIXED) ---
// Accepting props: isSoundOn and toggleSound
const Settings = ({ isDark, toggleTheme, isSoundOn, toggleSound }) => {
  return (
    <div className="fade-in">
      <header><h1>System Settings</h1></header>
      <div className="panel settings-panel">
        <div className="setting-item">
          <div className="setting-info">
            <h4>Dark Mode</h4>
            <p>Switch between light and dark themes</p>
          </div>
          <button className={`toggle-btn ${isDark ? 'active' : ''}`} onClick={toggleTheme}>
            {isDark ? 'ON' : 'OFF'}
          </button>
        </div>
        
        {/* FIXED SOUND BUTTON */}
        <div className="setting-item">
          <div className="setting-info">
            <h4>Notification Sound</h4>
            <p>Play alert sound on intrusion</p>
          </div>
          <button 
            className={`toggle-btn ${isSoundOn ? 'active' : ''}`} 
            onClick={toggleSound} 
          >
            {isSoundOn ? 'ON' : 'OFF'}
          </button>
        </div>

        <div className="setting-item">
          <div className="setting-info">
            <h4>Data Refresh Rate</h4>
            <p>Current: 1 Second</p>
          </div>
          <div className="badge">1000ms</div>
        </div>
      </div>
    </div>
  );
};

// --- MAIN LAYOUT APP ---
const App = () => {
  const [isDark, setIsDark] = useState(false);
  const toggleTheme = () => setIsDark(!isDark);

  // NEW: SOUND STATE
  const [isSoundOn, setIsSoundOn] = useState(true);
  const toggleSound = () => setIsSoundOn(!isSoundOn);

  return (
    <Router>
      <div className={`app-container ${isDark ? 'dark-mode' : 'light-mode'}`}>
        <aside className="sidebar">
          <div className="brand">🛡️ Intusion Detection System</div>
          <nav>
            <NavLink to="/" label="Dashboard" />
            <NavLink to="/history" label="Attack History" />
            <NavLink to="/performance" label="Model Performance" />
            <NavLink to="/settings" label="Settings" />
          </nav>
          <div className="footer-status">System Online</div>
        </aside>

        <main className="main-content">
          <Routes>
            {/* Pass Sound Props to Dashboard */}
            <Route path="/" element={<Dashboard isDark={isDark} isSoundOn={isSoundOn} />} />
            <Route path="/history" element={<History />} />
            <Route path="/performance" element={<Performance isDark={isDark} />} />
            {/* Pass Sound Props to Settings */}
            <Route path="/settings" element={
              <Settings 
                isDark={isDark} 
                toggleTheme={toggleTheme} 
                isSoundOn={isSoundOn} 
                toggleSound={toggleSound} 
              />
            } />
          </Routes>
        </main>
      </div>
    </Router>
  );
};

const NavLink = ({ to, label }) => {
  const location = useLocation();
  const isActive = location.pathname === to;
  return (
    <Link to={to} className={`nav-item ${isActive ? 'active' : ''}`}>
      {label}
    </Link>
  );
};

export default App;
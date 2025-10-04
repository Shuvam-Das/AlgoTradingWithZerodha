import React, { useEffect, useRef, useState } from 'react';
import Plotly from 'plotly.js-dist-min';

const WS_URL = 'ws://localhost:8000/ws/ticks'; // Update to your backend websocket URL

export default function App() {
  const chartRef = useRef(null);
  const [ticks, setTicks] = useState([]);

  useEffect(() => {
    // Initial dummy data for chart
    const initialData = [{
      x: [1, 2, 3, 4, 5],
      y: [100, 102, 101, 105, 107],
      type: 'scatter',
      mode: 'lines+markers',
      marker: { color: 'blue' },
      name: 'Price'
    }];
    Plotly.newPlot(chartRef.current, initialData, {
      title: 'Live Price Chart',
      paper_bgcolor: '#f8f8ff',
      plot_bgcolor: '#f0f8ff',
      font: { family: 'Montserrat, sans-serif', size: 16, color: '#222' },
    });
  }, []);

  useEffect(() => {
    // WebSocket for live ticks
    const ws = new window.WebSocket(WS_URL);
    ws.onmessage = (event) => {
      try {
        const data = JSON.parse(event.data);
        setTicks((prev) => [...prev, data]);
        // Update chart with new tick
        Plotly.extendTraces(chartRef.current, { y: [[data.price]], x: [[data.time]] }, [0]);
      } catch {}
    };
    ws.onerror = () => {};
    ws.onclose = () => {};
    return () => ws.close();
  }, []);

  return (
    <div style={{ fontFamily: 'Montserrat, sans-serif', background: 'linear-gradient(90deg,#e0eafc,#cfdef3)', minHeight: '100vh', padding: '2rem' }}>
      <h1 style={{ color: '#2a3f5f', textShadow: '1px 1px 2px #b2bec3' }}>AlgoTradingWithZerodha</h1>
      <p style={{ color: '#222', fontSize: '1.2rem' }}>Live TradingView-style chart with real-time tick updates.</p>
      <div ref={chartRef} style={{ width: '100%', maxWidth: 900, height: 500, margin: '2rem auto', borderRadius: 12, boxShadow: '0 4px 24px #b2bec3' }} />
      <div style={{ marginTop: '2rem', background: '#fff', borderRadius: 8, boxShadow: '0 2px 8px #b2bec3', padding: '1rem' }}>
        <h2 style={{ color: '#0984e3' }}>Recent Ticks</h2>
        <ul style={{ listStyle: 'none', padding: 0 }}>
          {ticks.slice(-10).reverse().map((tick, idx) => (
            <li key={idx} style={{ padding: '0.5rem 0', borderBottom: '1px solid #eee' }}>
              <strong>{tick.symbol}</strong> @ <span style={{ color: '#00b894' }}>{tick.price}</span> <span style={{ color: '#636e72' }}>{tick.time}</span>
            </li>
          ))}
        </ul>
      </div>
    </div>
  );
}

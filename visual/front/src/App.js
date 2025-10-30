import React, { useState, useEffect, useRef } from 'react';
import axios from 'axios';
import { Graph } from 'react-d3-graph';

function App() {
  const [graphData, setGraphData] = useState({ nodes: [], links: [] });
  const [newVertex, setNewVertex] = useState({ id: '', data: '' });
  const [newLink, setNewLink] = useState({ source: '', target: '' });
  const [kpis, setKpis] = useState({}); // State to store KPI data
  const [commentary, setCommentary] = useState([]);
  const commentaryRef = useRef(null);

  // Fetch the initial graph data and set up SSE
  useEffect(() => {
    fetchGraphData();

    const eventSource = new EventSource('http://localhost:3000/api/subscribe-graph-data');
    eventSource.onmessage = (event) => {
      const updatedGraphData = JSON.parse(event.data);

      const firstKey = Object.keys(updatedGraphData)[0];
      if (firstKey === 'nodes') {
        setGraphData(processGraphData(updatedGraphData));
        console.log('graph')
      } else if (firstKey === 'kpis') {
        updateKPIs(updatedGraphData.kpis)
        console.log(updatedGraphData)
      } else if (firstKey === 'text') {
        console.log(updatedGraphData)
        pushText(updatedGraphData.text)
      }
    };

    return () => {
      eventSource.close();
    };
  }, []);

  // Fetch current graph data from backend
  const fetchGraphData = () => {
    axios.get('http://localhost:3000/api/graph-data')
      .then(response => {
        setGraphData(processGraphData(response.data));
      })
      .catch(error => {
        console.error('Error fetching graph data', error);
      });
  };

  const updateKPIs = (newKpis) => {
    setKpis(newKpis);
  };

  let ttsBuffer = ''; // Buffer for aggregating text for TTS


  const pushText = (newText) => {
    setCommentary(prev => {
      if (prev.length === 0) {
        return [newText];
      }
      const updatedCommentary = [...prev];
      updatedCommentary[updatedCommentary.length - 1] += newText;
      return updatedCommentary;
    });

    setTimeout(() => {
      if (commentaryRef.current) {
        commentaryRef.current.scrollTop = commentaryRef.current.scrollHeight;
      }
    }, 100);

    // speakText(newText);
    // Aggregate text for TTS
    ttsBuffer += newText;
    if (ttsBuffer.includes('\n')) {
      speakText(ttsBuffer);
      ttsBuffer = ''; // Reset buffer after speaking
    }
  };

  // Process graph data to calculate node colors based on incoming links
  const processGraphData = (data) => {
    const incomingLinkCounts = data.links.reduce((acc, link) => {
      acc[link.target] = (acc[link.target] || 0) + 1;
      return acc;
    }, {});

    const maxIncomingLinks = Math.max(...Object.values(incomingLinkCounts), 1);

    const nodes = data.nodes.map(node => ({
      ...node,
      color: calculateNodeColor(incomingLinkCounts[node.id] || 0, maxIncomingLinks),
    }));

    return { nodes, links: data.links };
  };

  const calculateNodeColor = (incomingLinks, maxIncomingLinks) => {
    const darkness = Math.floor((incomingLinks / maxIncomingLinks) * 255);
    return `rgb(${255 - darkness}, ${255 - darkness}, ${255 - darkness})`; // Darker for more links
  };

  const speakText = (text) => {
    if ('speechSynthesis' in window) {
      const utterance = new SpeechSynthesisUtterance(text);
      utterance.lang = 'en-GB';
      utterance.rate = 1; // speed (0.1 - 10)
      utterance.pitch = 1; // pitch (0 - 2)
      speechSynthesis.speak(utterance);
    } else {
      console.warn('no tts');
    }
  };

  const displayKPIs = () => {
    return (
      <div style={{ padding: '10px', backgroundColor: '#f8f9fa', borderBottom: '1px solid #ddd' }}>
        <h3>KPI Dashboard</h3>
        <div style={{ display: 'flex', flexWrap: 'wrap', gap: '10px', width: '200px' }}>
          {Object.entries(kpis).map(([key, value]) => (
            <div
              key={key}
              style={{
                border: '1px solid #ddd',
                borderRadius: '5px',
                padding: '10px',
                textAlign: 'left',
                minWidth: '200px', // Minimum width for the card
                backgroundColor: '#fff',
                boxShadow: '0 2px 5px rgba(0,0,0,0.1)',
                flex: '1 1 30%', // allow flex to grow and shrink, with 30% width per card
                boxSizing: 'border-box',
              }}
            >
              <h4 style={{ margin: 0 }}>{key}</h4>
              {typeof value === 'object' && value !== null ? (
                <ul style={{ padding: 0, listStyleType: 'none', margin: 0 }}>
                  {Object.entries(value).map(([subKey, subValue]) => (
                    <li key={subKey}>
                      <strong>{subKey}:</strong> {subValue}
                    </li>
                  ))}
                </ul>
              ) : (
                <p style={{ margin: 0, fontSize: '1.5em', fontWeight: 'bold' }}>{value}</p>
              )}
            </div>
          ))}
        </div>
      </div>
    );
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'row', height: '100vh', margin: 0 }}>
      {/* KPI Dashboard */}
      {displayKPIs()}

      {/* Graph Visualization */}
      <div style={{ flex: 1, display: 'flex', justifyContent: 'right', alignItems: 'right', border: '2px solid' }}>
        <Graph
          id="graph-id"
          data={graphData}
          config={{
            directed: true,
            nodeHighlightBehavior: true,
            height: window.innerHeight,
            width: window.innerWidth - 740,
            linkHighlightBehavior: true,
            staticGraph: false,
            automaticRearrangeAfterDropNode: false,
            highlightOpacity: 1,
            highlightDegree: 1,
            panAndZoom: true,
            initialZoom: 0.6,
            node: {
              fontSize: 14,
              highlightFontSize: 14,
              highlightFontWeight: 'bold',
              size: 300,
            },
            link: {
              renderLabel: true,
              semanticStrokeWidth: true,
              highlightColor: 'black',
            },
            d3: {
              gravity: -400,
              linkLength: 150,
              linkStrength: 2,
              charge: -400,
            },
          }}
        />
      </div>
      <div
        ref={commentaryRef}
        style={{
          width: '700px',
          maxHeight: '100%',
          overflowY: 'auto',
          padding: '10px',
          borderTop: '1px solid #ddd',
          backgroundColor: '#f0f0f0',
          wordWrap: 'break-word',
          whiteSpace: 'pre-wrap', // line breaks in long text
        }}
      >
        <h4>Commentary</h4>
        <p style={{ margin: '5px 0' }}>{commentary.join('')}</p>
      </div>

    </div>
  );
}

export default App;

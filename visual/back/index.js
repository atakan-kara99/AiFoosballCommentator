const express = require('express');
const cors = require('cors');

const app = express();
app.use(express.json());
app.use(cors());

// Graph structure
let graph = {
  nodes: [],
  links: [],
  addNode(id, data, x = 500, y = 500) {
    const existingNodeIndex = this.nodes.findIndex((node) => node.id === id);
    if (existingNodeIndex !== -1) {
      this.nodes[existingNodeIndex] = { id, data, x, y }; // Override existing node
    } else {
      this.nodes.push({ id, data, x, y });
    }
  },
  addLink(source, target) {
    const existingLinkIndex = this.links.findIndex(
      (link) => link.source === source && link.target === target
    );
    if (existingLinkIndex !== -1) {
      this.links[existingLinkIndex] = { source, target }; // Override existing link
    } else {
      this.links.push({ source, target });
    }
  },
};

// KPI structure
let kpi = {
  kpis: {},
  add(key, value) {
    this.kpis[key] = value;
  },
  clear_all() {
    this.kpis = {};
  }
}

// Enpoint to receive KPI changes.
app.post('/kpi/push', (req, res) => {
  const b = req.body;
  kpi.add(b.id, b.value);
  res.send({ success: true });

  sendKPIs();
});

// Endpoint to reset the graph.
app.post('/api/reset', (req, res) => {
  graph.nodes = [];
  graph.links = [];
  // kpi.clear_all();

  res.send({ success: true });
  sendGraphDataToClients();
  sendKPIs()
  console.log('Reset')
});

// Endpoint to get the current graph.
app.get('/api/graph-data', (req, res) => {
  res.json(graph);
});

// Endpoint to add a vertex (node)
app.post('/api/vertices', (req, res) => {
  const newVertex = req.body;
  graph.addNode(newVertex.id, newVertex.data);
  res.status(201).send({ success: true });

  // Notify frontend to update
  sendGraphDataToClients();
});

// Endpoint to add a link (edge)
app.post('/api/links', (req, res) => {
  const newLink = req.body;
  graph.addLink(newLink.source, newLink.target);
  res.status(201).send({ success: true });

  // Notify frontend to update
  sendGraphDataToClients();
});

// Endpoint to push commentary text.
app.post('/pushText', (req, res) => {
  console.log(req.body)
  const data = JSON.stringify(req.body);
  sseClients.forEach((client) => client.write(`data: ${data}\n\n`));
  res.status(201).send({ success: true });
});


// SSE Setup for clients to listen for graph and KPI data.
const sseClients = [];
app.get('/api/subscribe-graph-data', (req, res) => {
  res.setHeader('Content-Type', 'text/event-stream');
  res.setHeader('Cache-Control', 'no-cache');
  res.setHeader('Connection', 'keep-alive');

  sseClients.push(res);
  console.log('Client connected for SSE');

  req.on('close', () => {
    console.log('Client disconnected');
    sseClients.splice(sseClients.indexOf(res), 1);
  });
});

// Function to send graph data to all connected clients
const sendGraphDataToClients = () => {
  const data = JSON.stringify(graph);
  sseClients.forEach((client) => client.write(`data: ${data}\n\n`));
};

// Function to send KPI data to all connected clients
const sendKPIs = () => {
  const data = JSON.stringify(kpi);
  sseClients.forEach((client) => client.write(`data: ${data}\n\n`));
};

// Start the server
app.listen(3000, () => {
  console.log('Backend running on http://localhost:3000');
});


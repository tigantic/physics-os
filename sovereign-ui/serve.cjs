/**
 * Production-like static server with API reverse proxy.
 *
 * Serves the SvelteKit static build from ./build and proxies
 * /api/* and /ws requests to the Python backend on port 8420.
 *
 * Usage: node serve.cjs [--port 4000]
 */

const http = require('http');
const fs = require('fs');
const path = require('path');

const BACKEND = 'http://127.0.0.1:8421';
const API_KEY = process.env.FPS_API_KEY;
if (!API_KEY) {
  console.error('FATAL: FPS_API_KEY environment variable is required.\n  Export it before starting: export FPS_API_KEY=<your-key>');
  process.exit(1);
}
const BUILD_DIR = path.resolve(__dirname, 'build');
const PORT = parseInt(process.argv.find((_, i, a) => a[i - 1] === '--port') || '4000', 10);

const MIME_TYPES = {
  '.html': 'text/html; charset=utf-8',
  '.js': 'application/javascript; charset=utf-8',
  '.css': 'text/css; charset=utf-8',
  '.json': 'application/json; charset=utf-8',
  '.svg': 'image/svg+xml',
  '.png': 'image/png',
  '.jpg': 'image/jpeg',
  '.woff2': 'font/woff2',
  '.woff': 'font/woff',
  '.ttf': 'font/ttf',
  '.ico': 'image/x-icon',
};

function proxyRequest(clientReq, clientRes) {
  const url = new URL(clientReq.url, BACKEND);

  const proxyHeaders = { ...clientReq.headers };
  proxyHeaders['host'] = url.host;
  proxyHeaders['x-api-key'] = API_KEY;
  delete proxyHeaders['origin'];

  const proxyReq = http.request(
    {
      hostname: url.hostname,
      port: url.port,
      path: url.pathname + url.search,
      method: clientReq.method,
      headers: proxyHeaders,
    },
    (proxyRes) => {
      clientRes.writeHead(proxyRes.statusCode, proxyRes.headers);
      proxyRes.pipe(clientRes, { end: true });
    },
  );

  proxyReq.on('error', (err) => {
    console.error(`[proxy] ${clientReq.method} ${clientReq.url} → ${err.message}`);
    clientRes.writeHead(502, { 'Content-Type': 'application/json' });
    clientRes.end(JSON.stringify({ error: 'Backend unavailable' }));
  });

  clientReq.pipe(proxyReq, { end: true });
}

function serveStatic(req, res) {
  let filePath = path.join(BUILD_DIR, req.url === '/' ? 'index.html' : req.url);

  // Security: prevent directory traversal
  if (!filePath.startsWith(BUILD_DIR)) {
    res.writeHead(403);
    res.end('Forbidden');
    return;
  }

  fs.stat(filePath, (err, stats) => {
    if (err || !stats.isFile()) {
      // SPA fallback: serve index.html for any non-file route
      filePath = path.join(BUILD_DIR, 'index.html');
    }

    fs.readFile(filePath, (readErr, data) => {
      if (readErr) {
        res.writeHead(500);
        res.end('Internal Server Error');
        return;
      }

      const ext = path.extname(filePath).toLowerCase();
      const contentType = MIME_TYPES[ext] || 'application/octet-stream';

      // Cache immutable assets aggressively
      const cacheControl = filePath.includes('/_app/immutable/')
        ? 'public, max-age=31536000, immutable'
        : 'no-cache';

      res.writeHead(200, {
        'Content-Type': contentType,
        'Content-Length': data.length,
        'Cache-Control': cacheControl,
      });
      res.end(data);
    });
  });
}

const server = http.createServer((req, res) => {
  const url = req.url;

  // Log every request for debugging
  const ts = new Date().toISOString().slice(11, 23);
  console.log(`[${ts}] ${req.method} ${url}`);

  // Proxy API and WebSocket requests to backend
  if (url.startsWith('/api/') || url.startsWith('/api') || url === '/ws') {
    proxyRequest(req, res);
    return;
  }

  // Health check for the serving layer itself
  if (url === '/health') {
    res.writeHead(200, { 'Content-Type': 'application/json' });
    res.end(JSON.stringify({ status: 'serving', port: PORT }));
    return;
  }

  // Serve static files
  serveStatic(req, res);
});

server.listen(PORT, '0.0.0.0', () => {
  console.log(`\n  Sovereign UI — Production Server`);
  console.log(`  ─────────────────────────────────`);
  console.log(`  Local:   http://localhost:${PORT}/`);
  console.log(`  Network: http://0.0.0.0:${PORT}/`);
  console.log(`  Backend: ${BACKEND}`);
  console.log(`  Static:  ${BUILD_DIR}\n`);
});

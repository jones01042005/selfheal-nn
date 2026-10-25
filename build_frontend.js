const fs = require('node:fs');
const path = require('node:path');

const backendUrl = (process.env.BACKEND_URL || '').trim().replace(/\/+$/, '');
if (!backendUrl) {
    throw new Error('Set BACKEND_URL in the Vercel project build environment.');
}

let parsedBackendUrl;
try {
    parsedBackendUrl = new URL(backendUrl);
} catch {
    throw new Error('BACKEND_URL must be an absolute HTTP or HTTPS URL.');
}
if (parsedBackendUrl.protocol !== 'http:' && parsedBackendUrl.protocol !== 'https:') {
    throw new Error('BACKEND_URL must use HTTP or HTTPS.');
}

const outputDirectory = path.join(__dirname, 'dist');
fs.mkdirSync(outputDirectory, { recursive: true });
fs.copyFileSync(
    path.join(__dirname, 'static', 'index.html'),
    path.join(outputDirectory, 'index.html')
);
fs.cpSync(
    path.join(__dirname, 'static'),
    path.join(outputDirectory, 'static'),
    { recursive: true }
);
fs.writeFileSync(
    path.join(outputDirectory, 'static', 'config.js'),
    `window.BACKEND_URL = ${JSON.stringify(backendUrl)};\n`
);

console.log(`Built static frontend for ${parsedBackendUrl.origin}`);

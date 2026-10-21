// Preload (node --require) that makes every network access fail and records it in the file
// named by NETWORK_LOG: sockets (net, tls, http, https and undici all connect through
// net.Socket), DNS lookups and fetch.
'use strict';

const dns = require('node:dns');
const fs = require('node:fs');
const net = require('node:net');

const attempts = [];
function blocked(kind, detail) {
  attempts.push(`${kind} ${detail}`);
  const error = new Error(`network access blocked: ${kind} ${detail}`);
  error.code = 'ENETBLOCKED';
  return error;
}

net.Socket.prototype.connect = function connect(...args) {
  throw blocked('socket.connect', JSON.stringify(args[0]));
};
dns.lookup = (host) => { throw blocked('dns.lookup', host); };
dns.promises.lookup = async (host) => { throw blocked('dns.promises.lookup', host); };
globalThis.fetch = async (url) => { throw blocked('fetch', String(url)); };

process.on('exit', () => fs.writeFileSync(process.env.NETWORK_LOG, JSON.stringify(attempts)));

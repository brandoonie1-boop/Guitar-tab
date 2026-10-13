// Saved tabs on this device (IndexedDB). Every call tolerates storage being unavailable.
const DB_NAME = 'guitar-tab';
const DB_VER = 1;
let dbp = null;

function open() {
  if (dbp) return dbp;
  dbp = new Promise((resolve) => {
    try {
      const req = indexedDB.open(DB_NAME, DB_VER);
      req.onupgradeneeded = () => {
        const db = req.result;
        if (!db.objectStoreNames.contains('projects')) db.createObjectStore('projects', { keyPath: 'id' });
        if (!db.objectStoreNames.contains('blobs')) db.createObjectStore('blobs');
      };
      req.onsuccess = () => resolve(req.result);
      req.onerror = () => resolve(null);
      req.onblocked = () => resolve(null);
    } catch { resolve(null); }
  });
  return dbp;
}

function tx(db, store, mode, fn) {
  return new Promise((resolve) => {
    try {
      const t = db.transaction(store, mode);
      const st = t.objectStore(store);
      const r = fn(st);
      t.oncomplete = () => resolve(r && 'result' in r ? r.result : true);
      t.onerror = () => resolve(null);
      t.onabort = () => resolve(null);
    } catch { resolve(null); }
  });
}

export async function saveProject(p) {
  const db = await open();
  if (!db) return false;
  return tx(db, 'projects', 'readwrite', (s) => s.put(p));
}

export async function listProjects() {
  const db = await open();
  if (!db) return [];
  const all = await tx(db, 'projects', 'readonly', (s) => s.getAll());
  return (all || []).map(({ id, name, created, duration, summary }) => ({ id, name, created, duration, summary })).sort((a, b) => b.created - a.created);
}

export async function getProject(id) {
  const db = await open();
  if (!db) return null;
  return tx(db, 'projects', 'readonly', (s) => s.get(id));
}

export async function saveBlob(id, name, blob) {
  const db = await open();
  if (!db) return false;
  return tx(db, 'blobs', 'readwrite', (s) => s.put(blob, `${id}:${name}`));
}

export async function getBlob(id, name) {
  const db = await open();
  if (!db) return null;
  return tx(db, 'blobs', 'readonly', (s) => s.get(`${id}:${name}`));
}

export async function deleteProject(id) {
  const db = await open();
  if (!db) return false;
  await tx(db, 'projects', 'readwrite', (s) => s.delete(id));
  await tx(db, 'blobs', 'readwrite', (s) => s.delete(IDBKeyRange.bound(`${id}:`, `${id}:￿`)));
  return true;
}

export async function storageEstimate() {
  try { return await navigator.storage.estimate(); } catch { return null; }
}

export async function persist() {
  try { if (navigator.storage && navigator.storage.persist) return await navigator.storage.persist(); } catch { /* ignore */ }
  return false;
}

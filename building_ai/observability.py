"""Small SQLite-backed, redacted Agent trace store."""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
def redact(value): return re.sub(r'(?i)(api[_ -]?key|authorization|password|token)\s*[:=]\s*[^,\s]+',r'\1=[REDACTED]',str(value))
class TraceStore:
 def __init__(self,database):
  self.database=database
  with database.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS agent_traces (trace_id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
 def save(self,payload):
  payload={**payload,'created_at':datetime.now(timezone.utc).isoformat()}; payload['query']=redact(payload.get('query',''))
  with self.database.connect() as c:c.execute('INSERT OR REPLACE INTO agent_traces VALUES (?,?)',(payload['trace_id'],json.dumps(payload,ensure_ascii=False,default=str)))
 def get(self,trace_id):
  with self.database.connect() as c:r=c.execute('SELECT payload FROM agent_traces WHERE trace_id=?',(trace_id,)).fetchone()
  return json.loads(r['payload']) if r else None
 def list(self):
  with self.database.connect() as c: rows=c.execute('SELECT payload FROM agent_traces ORDER BY rowid DESC LIMIT 50').fetchall()
  return [json.loads(r['payload']) for r in rows]

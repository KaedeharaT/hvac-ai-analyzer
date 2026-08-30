"""Small SQLite-backed, redacted Agent trace store."""
from __future__ import annotations
import json, re
from datetime import datetime, timezone
_SECRET_KEY=re.compile(r'(?i)(?:api[_ -]?key|authorization|password|(?:access[_ -]?)?token|secret|credential)s?$')
_SECRET_VALUE=re.compile(r'(?i)\b(api[_ -]?key|authorization|password|(?:access[_ -]?)?token|secret|credential)\b\s*[:=]\s*(?:bearer\s+[^,\s}\]]+|"[^"]*"|\'[^\']*\'|[^,\s}\]]+)')
_BEARER=re.compile(r'(?i)\bbearer\s+[^,\s}\]]+')
def redact(value):
 if isinstance(value,dict): return {key:('[REDACTED]' if _SECRET_KEY.fullmatch(str(key)) else redact(item)) for key,item in value.items()}
 if isinstance(value,list): return [redact(item) for item in value]
 if isinstance(value,tuple): return tuple(redact(item) for item in value)
 if not isinstance(value,str): return value
 text=_SECRET_VALUE.sub(lambda match:f'{match.group(1)}=[REDACTED]',value)
 return _BEARER.sub('Bearer [REDACTED]',text)
class TraceStore:
 def __init__(self,database):
  self.database=database
  with database.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS agent_traces (trace_id TEXT PRIMARY KEY,payload TEXT NOT NULL)')
 def save(self,payload):
  payload=redact({**payload,'created_at':datetime.now(timezone.utc).isoformat()})
  with self.database.connect() as c:c.execute('INSERT OR REPLACE INTO agent_traces VALUES (?,?)',(payload['trace_id'],json.dumps(payload,ensure_ascii=False,default=str)))
 def get(self,trace_id):
  with self.database.connect() as c:r=c.execute('SELECT payload FROM agent_traces WHERE trace_id=?',(trace_id,)).fetchone()
  return json.loads(r['payload']) if r else None
 def list(self):
  with self.database.connect() as c: rows=c.execute('SELECT payload FROM agent_traces ORDER BY rowid DESC LIMIT 50').fetchall()
  return [json.loads(r['payload']) for r in rows]

"""Scoped persistent conversation and project memory; never cross-project by default."""
from __future__ import annotations
import json
class MemoryStore:
 def __init__(self,database):
  self.database=database
  with database.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS agent_memory (project_id TEXT, conversation_id TEXT, kind TEXT, key TEXT, payload TEXT, PRIMARY KEY(project_id,conversation_id,kind,key))')
 def put(self,project_id,conversation_id,kind,key,value):
  with self.database.connect() as c:c.execute('INSERT OR REPLACE INTO agent_memory VALUES (?,?,?,?,?)',(project_id,conversation_id,kind,key,json.dumps(value,ensure_ascii=False)))
 def get(self,project_id,conversation_id,kind,key):
  with self.database.connect() as c:r=c.execute('SELECT payload FROM agent_memory WHERE project_id=? AND conversation_id=? AND kind=? AND key=?',(project_id,conversation_id,kind,key)).fetchone()
  return json.loads(r['payload']) if r else None
 def delete(self,project_id,conversation_id,kind,key):
  """Remove a user-cleared, conversation-scoped focus value."""
  with self.database.connect() as c:c.execute('DELETE FROM agent_memory WHERE project_id=? AND conversation_id=? AND kind=? AND key=?',(project_id,conversation_id,kind,key))
 def put_project(self,project_id,key,value):
  """Persist project facts separately from a user conversation."""
  self.put(project_id,'__project__','project',key,value)
 def get_project(self,project_id,key):
  return self.get(project_id,'__project__','project',key)

"""Local keyword knowledge retrieval with source-aware chunks."""
from __future__ import annotations
import re,uuid
class KnowledgeService:
 def __init__(self,database):
  self.database=database
  with database.connect() as c:c.execute('CREATE TABLE IF NOT EXISTS knowledge_chunks (chunk_id TEXT PRIMARY KEY,title TEXT,source TEXT,section TEXT,text TEXT,metadata TEXT)')
 def ingest(self,title,source,text,section='General',metadata='{}'):
  cid=str(uuid.uuid4())
  with self.database.connect() as c:c.execute('INSERT INTO knowledge_chunks VALUES (?,?,?,?,?,?)',(cid,title,source,section,text,metadata))
  return cid
 def search(self,query,top_k=3):
  words=set(re.findall(r'\w+',query.casefold()))
  with self.database.connect() as c: rows=c.execute('SELECT * FROM knowledge_chunks').fetchall()
  scored=[]
  for r in rows:
   score=len(words & set(re.findall(r'\w+',r['text'].casefold())))
   if score: scored.append({'chunk_id':r['chunk_id'],'title':r['title'],'source':r['source'],'section':r['section'],'text':r['text'],'score':score})
  return sorted(scored,key=lambda x:x['score'],reverse=True)[:top_k]

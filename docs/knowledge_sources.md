# BuildingAI knowledge sources

BuildingAI ships a small, attributable engineering corpus rather than a general
PDF-chat database. Project data and formal diagnostics establish what is true at
a site; the knowledge corpus explains concepts and provides bounded checks.

Versioned knowledge assets:

- Registry: [`knowledge/source_registry.json`](../knowledge/source_registry.json)
- Curated records: [`knowledge/curated/`](../knowledge/curated/)
- Portable chunks: [`knowledge/chunks/knowledge_chunks.jsonl`](../knowledge/chunks/knowledge_chunks.jsonl)
- Alias metadata: [`knowledge/metadata/`](../knowledge/metadata/)
- Keyword/CJK index: [`knowledge/index/keyword_cjk_index.json`](../knowledge/index/keyword_cjk_index.json)

The corpus contains original, factual summaries and structured ontology facts.
It never stores full paywalled standards, vendor manuals, user documents, or
downloaded web pages.

| Country | Organization | Source | Category | Official URL | Language | Usage / license status | Ingest strategy |
| --- | --- | --- | --- | --- | --- | --- | --- |
| US | Project Haystack | Documentation | Semantic / points | [docs](https://www.project-haystack.org/doc/index) | English | AFL-3.0; structured facts | `OPEN_FULLTEXT` |
| US | Brick Schema | Relationships | Ontology / topology | [docs](https://docs.brickschema.org/brick/relationships.html) | English | Public ontology summaries | `OPEN_FULLTEXT` |
| US | Brick / ASHRAE 223 public docs | Connections overview | Topology | [docs](https://reconciliation.brickschema.org/modeling/connections.html) | English | No paid ASHRAE text | `PUBLIC_SUMMARY` |
| US | DOE / NREL | EnergyPlus Engineering Reference | Chiller / COP / load | [reference](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v26.1.0/EngineeringReference.pdf) | English | Official factual summaries | `PUBLIC_SUMMARY` |
| US | DOE FEMP | O&M Best Practices | O&M / commissioning | [guidance](https://www.energy.gov/cmei/femp/articles/operations-and-maintenance-best-practices-guide-achieving-operational-efficiency) | English | Government public guidance | `PUBLIC_SUMMARY` |
| US | DOE Better Buildings | Process Cooling and HVAC | Pump / chiller / retrofit | [guidance](https://betterbuildingssolutioncenter.energy.gov/better-plants/process-cooling-and-hvac) | English | Government public guidance | `PUBLIC_SUMMARY` |
| US | DOE FEMP / PNNL | O&M Challenges and Solutions | Re-tuning / controls | [guidance](https://www.energy.gov/cmei/femp/operations-and-maintenance-challenges-and-solutions) | English | Government public guidance | `PUBLIC_SUMMARY` |
| US | NREL | Building Component Library | Retrofit measures | [BCL](https://bcl.nrel.gov/) | English | Metadata only | `METADATA_ONLY` |
| China | Ministry of Ecology and Environment | 公共机构节能条例 | Metering / O&M | [regulation](https://www.mee.gov.cn/zcwj/gwywj/202001/t20200115_759435.shtml) | 中文 | Government factual summaries | `PUBLIC_SUMMARY` |
| China | NDRC | 建筑全过程节能降碳管理 | Retrofit / operation | [guidance](https://www.ndrc.gov.cn/xxgk/jd/jd/202403/t20240321_1365112_ext.html) | 中文 | Government public guidance | `PUBLIC_SUMMARY` |
| China | NDRC | 中国“双十佳”最佳节能技术和实践 | BEMS / controls | [practice](https://www.ndrc.gov.cn/xxgk/zcfb/gg/201601/W020190905485554900548.pdf) | 中文 | Government public practice | `PUBLIC_SUMMARY` |
| China | SAMR | 公共机构办公区节能运行管理规范 | O&M standard | [registry](https://std.samr.gov.cn/gb/search/gbDetailed?id=e7m%2Bs2%2BsfbU%3D&mode=p) | 中文 | Metadata only; no full standard | `METADATA_ONLY` |
| China | MOHURD / NDRC | 建筑能效与绿色发展 | Existing-building retrofit | [guidance](https://www.ndrc.gov.cn/xwdt/ztzl/2022qgjnxcz/bmjncx/202206/t20220612_1327158_ext.html) | 中文 | Government public guidance | `PUBLIC_SUMMARY` |
| Japan | METI Agency for Natural Resources and Energy | ZEB guidance | ZEB / retrofit | [guidance](https://www.enecho.meti.go.jp/category/saving_and_new/saving/enterprise/support/index02.html) | 日本語 | Government public guidance | `PUBLIC_SUMMARY` |
| Japan | Ministry of the Environment | ZEBを実現するための技術 | HVAC / control | [ZEB Portal](https://www.env.go.jp/earth/zeb/detail/06.html) | 日本語 | Government public guidance | `PUBLIC_SUMMARY` |
| Japan | Ministry of the Environment | 特に既存改修ZEB化の場合 | Retrofit | [ZEB Portal](https://www.env.go.jp/earth/zeb/detail/12.html) | 日本語 | Government public guidance | `PUBLIC_SUMMARY` |
| Japan | MLIT | 省エネ基準引き上げへ。脱炭素化も。 | Building energy | [guidance](https://www.mlit.go.jp/shoene-jutaku/index.html) | 日本語 | Government public guidance | `PUBLIC_SUMMARY` |
| Japan | Building Research Institute / NILIM | 建築物のエネルギー消費性能に関する技術情報 | BEMS / energy analysis | [resource](https://www.kenken.go.jp/becc/) | 日本語 | Public technical information | `PUBLIC_SUMMARY` |
| Global | BuildingAI | Engineering review checklist | ΔT / evidence | Local project catalog | Multilingual | Original guidance; never project evidence | `OPEN_FULLTEXT` |

## Rebuild

```powershell
python scripts/build_knowledge_base.py
```

The command deterministically regenerates all curated JSONL, metadata, and
keyword/CJK index files, then ingests the same chunks into the configured local
SQLite knowledge storage. A separate database can be validated without touching
user data:

```powershell
python scripts/build_knowledge_base.py --database .\knowledge-validation.sqlite3
```

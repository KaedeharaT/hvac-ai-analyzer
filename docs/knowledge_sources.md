# BuildingAI knowledge sources

BuildingAI's knowledge layer is a small, attributable engineering reference for
semantic interpretation and user-facing operational guidance. It is **not** a
general PDF dump and it never establishes what is happening in a project. Project
data and formal diagnostic tools establish project facts; this catalog explains
concepts and suggests evidence to inspect.

The machine-readable registry is [`knowledge/source_registry.json`](../knowledge/source_registry.json).
The builder ingests original, short factual summaries into the configured local
SQLite database; it does not download or redistribute source manuals.

| Country | Organization | Source | Category | URL | Use and status | Strategy |
| --- | --- | --- | --- | --- | --- | --- |
| US | Project Haystack | Documentation | Semantic / points | [official docs](https://project-haystack.org/doc/index) | AFL-3.0 documentation; attributed structured facts | `OPEN_FULLTEXT` |
| US | Brick Schema | Relationships | Ontology / topology | [official docs](https://docs.brickschema.org/brick/relationships.html) | Public ontology documentation; short summaries | `OPEN_FULLTEXT` |
| US | Brick / ASHRAE 223 public docs | Connections overview | System topology | [official docs](https://reconciliation.brickschema.org/modeling/connections.html) | Public overview only; no paid ASHRAE text | `PUBLIC_SUMMARY` |
| US | DOE / NREL | EnergyPlus Engineering Reference | Chiller / COP / load | [official reference](https://energyplus.net/assets/nrel_custom/pdfs/pdfs_v24.2.0/EngineeringReference.pdf) | Short factual engineering summaries | `PUBLIC_SUMMARY` |
| US | DOE FEMP | Operations and Maintenance Best Practices | O&M / commissioning | [official guidance](https://www.energy.gov/cmei/femp/articles/operations-and-maintenance-best-practices-guide-achieving-operational-efficiency) | Government public guidance; short summaries | `PUBLIC_SUMMARY` |
| US | DOE Better Buildings | Process Cooling and HVAC | Pump / chiller / retrofit | [official guidance](https://betterbuildingssolutioncenter.energy.gov/better-plants/process-cooling-and-hvac) | Government public guidance; short summaries | `PUBLIC_SUMMARY` |
| US | DOE FEMP / PNNL | O&M Challenges and Solutions | Re-tuning / controls | [official guidance](https://www.energy.gov/cmei/femp/operations-and-maintenance-challenges-and-solutions) | Government public guidance; short summaries | `PUBLIC_SUMMARY` |
| US | NREL | Building Component Library | Retrofit measures | [BCL](https://bcl.nrel.gov/) | Registry metadata only; measure terms need individual review | `METADATA_ONLY` |
| China | Ministry of Ecology and Environment | 公共机构节能条例 | Metering / O&M | [official regulation](https://www.mee.gov.cn/zcwj/gwywj/202001/t20200115_759435.shtml) | Narrow factual operational summaries | `PUBLIC_SUMMARY` |
| China | National Development and Reform Commission | 加强建筑全过程节能降碳管理 | Retrofit / operation | [official guidance](https://www.ndrc.gov.cn/xxgk/jd/jd/202403/t20240321_1365112_ext.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| China | NDRC | 中国“双十佳”最佳节能技术和实践 | BEMS / controls | [official practice](https://www.ndrc.gov.cn/xxgk/zcfb/gg/201601/W020190905485554900548.pdf) | Short attributed summaries | `PUBLIC_SUMMARY` |
| China | SAMR | 公共机构办公区节能运行管理规范 | O&M standard | [standard registry](https://std.samr.gov.cn/gb/search/gbDetailed?id=e7m%2Bs2%2BsfbU%3D&mode=p) | Standard metadata only; no full standard text | `METADATA_ONLY` |
| China | MOHURD / NDRC | 建筑能效与绿色发展 guidance | Existing-building retrofit | [official guidance](https://www.ndrc.gov.cn/xwdt/ztzl/2022qgjnxcz/bmjncx/202206/t20220612_1327158_ext.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Japan | METI Agency for Natural Resources and Energy | ZEB support and design guidance | ZEB / retrofit | [official guidance](https://www.enecho.meti.go.jp/category/saving_and_new/saving/enterprise/support/index02.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Japan | Ministry of the Environment | ZEBを実現するための技術 | HVAC / control | [ZEB Portal](https://www.env.go.jp/earth/zeb/detail/06.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Japan | Ministry of the Environment | 特に既存改修ZEB化の場合 | Retrofit | [ZEB Portal](https://www.env.go.jp/earth/zeb/detail/12.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Japan | MLIT | 省エネ基準引き上げへ。脱炭素化も。 | Building energy | [official guidance](https://www.mlit.go.jp/shoene-jutaku/index.html) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Japan | Building Research Institute / NILIM | 建築物のエネルギー消費性能に関する技術情報 | BEMS / energy analysis | [official resource](https://www.kenken.go.jp/becc/) | Short attributed summaries | `PUBLIC_SUMMARY` |
| Global | BuildingAI | Engineering review checklist | ΔT / evidence collection | [project repository](https://github.com/your-org/building-ai-desktop) | Original guidance; never project evidence | `OPEN_FULLTEXT` |

## Rebuild locally

```powershell
python scripts/build_knowledge_base.py
```

The command is idempotent and writes to the configured user-local BuildingAI
SQLite database. To validate a separate database without touching user data:

```powershell
python scripts/build_knowledge_base.py --database .\knowledge-validation.sqlite3
```

Only the source registry and compact original summaries are versioned. Do not add
paywalled standards, vendor manuals, large PDFs, user documents, or copied web
sites. Users may ingest documents they are authorized to use through a future
user-supplied-document workflow.

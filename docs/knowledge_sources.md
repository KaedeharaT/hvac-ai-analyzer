# BuildingAI Knowledge Sources

The runtime stores source URL/usage metadata with every chunk.  Official public
sources are routed by purpose: Project Haystack (semantic tags), Brick Schema
(equipment/point relationships), Open223 (system topology), DOE FEMP (O&M),
NREL BCL and DOE Better Buildings (energy measures), and EnergyPlus Engineering
Reference (engineering explanations).  The automated test corpus uses only the
locally authored **BuildingAI HVAC Operations Guide**; it does not copy external
documents.

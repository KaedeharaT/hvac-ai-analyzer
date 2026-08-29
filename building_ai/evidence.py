from enum import Enum
class EvidenceStatus(str,Enum): SUFFICIENT='SUFFICIENT'; PARTIAL='PARTIAL'; INSUFFICIENT='INSUFFICIENT'
def check_reasoning(evidence:dict, requires_cause:bool=False):
    if not evidence: return EvidenceStatus.INSUFFICIENT
    if requires_cause and not ('get_diagnostic_findings' in evidence and 'get_energy_timeseries' in evidence): return EvidenceStatus.PARTIAL
    return EvidenceStatus.SUFFICIENT

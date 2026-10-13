"""Plain-language rendering of the same read model for desktop and Agent."""
from building_ai.i18n.manager import TABLES, normalize_language


def render_review(report: dict, language: str) -> str:
    table = TABLES[normalize_language(language)]
    def t(key, **values): return table.get(key, key).format(**values)
    stages = report['stages']
    lines = [t('retrofit_title'), t('retrofit_workflow'), '',
             t('retrofit_stage_status', objects=stages['drawing_objects'], links=stages['verified_associations'], actions=stages['supported_operating_actions']),
             t('retrofit_linked' if report['workflow_status']=='evidence_linked' else 'retrofit_partial'),
             t('retrofit_boundary'), '', t('retrofit_layer_operation')]
    if not stages['bems_available']: lines.append(t('retrofit_need_bems'))
    if not stages['diagnosis_completed']: lines.append(t('retrofit_need_analysis'))
    items = report['layers']['operation']['items']
    if not items: lines.append(t('retrofit_no_operations'))
    for item in items:
        prefix = 'guidance_' + item['finding_type']
        if prefix+'_problem' not in table: prefix = 'guidance_other'
        lines += [f"{item['equipment_name']} — {t(prefix+'_problem')}", t('retrofit_bems_origin'), t(prefix+'_explanation')]
        for n in range(10):
            key=f'{prefix}_action_{n}'
            if key not in table: break
            lines.append(f"  {n+1}. {t(key)}")
        lines.append(t('retrofit_finding_reference', ids=', '.join(item['finding_ids'])))
    if report['bems_evidence']['unavailable_equipment']:
        lines.append(t('retrofit_missing_inputs', names=', '.join(x['equipment_name'] for x in report['bems_evidence']['unavailable_equipment'])))
    lines += ['', t('retrofit_layer_equipment'), t('retrofit_general_origin'), t('retrofit_equipment_checklist'),
              '', t('retrofit_layer_building'), t('retrofit_general_origin'), t('retrofit_building_checklist'),
              '', t('retrofit_drawing_context')]
    classes=report['drawing_evidence']['confirmed_classes']
    lines.append(t('retrofit_object_counts', aircon=classes.get('aircon',0), windows=classes.get('window',0), baseline=classes.get('baseline_mark',0)))
    for row in report['drawing_evidence']['associations']:
        lines.append(f"{row['equipment_name']} · {row['object_id']} · " + t('retrofit_association_'+row['association_review']['association_status']))
    lines.append(t('retrofit_spatial_boundary'))
    if report['drawing_evidence']['invalid_drawings']: lines.append(t('retrofit_invalid_source'))
    return '\n'.join(lines)

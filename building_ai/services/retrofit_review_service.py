"""Read model joining existing evidence; no new engineering calculations or rules."""
from __future__ import annotations
from collections import Counter
import json


class RetrofitReviewService:
    def __init__(self, drawings):
        self.drawings = drawings

    def build(self, state: dict, equipment_name: str | None = None) -> dict:
        project = state['project']
        organization = state.get('equipment')
        equipment = list(organization.equipment) if organization else []
        selected = [e for e in equipment if equipment_name is None or e.name == equipment_name or e.equipment_id == equipment_name]
        ids = {e.equipment_id for e in selected}
        names = {e.equipment_id:e.name for e in selected}
        drawings = self.drawings.list_drawings(project.project_id) if self.drawings else []
        detections = self.drawings.list_detections(project.project_id) if self.drawings else []
        confirmed = [r for r in detections if r['review_status']=='confirmed']
        links, verified_ids, invalid_drawings = [], set(), []
        for drawing in drawings:
            payload = json.loads(drawing.get('payload_json') or '{}')
            try:
                self.drawings.image_positions(project.project_id,drawing['drawing_id'])
            except (ValueError, OSError):
                invalid_drawings.append(drawing['drawing_id']); continue
            for row in confirmed:
                if row['drawing_id'] != drawing['drawing_id'] or row['equipment_id'] not in ids:
                    continue
                review = payload.get('associations',{}).get(row['detection_id'], {'association_status':'pending'})
                verified = (review.get('association_status')=='verified' and bool(review.get('reviewer'))
                            and bool(review.get('evidence_note')) and review.get('equipment_id')==row['equipment_id'])
                if verified: verified_ids.add(row['equipment_id'])
                links.append({'drawing_id':drawing['drawing_id'],'object_id':row['detection_id'],
                    'source_image_hash':payload.get('source_image_hash'),'equipment_id':row['equipment_id'],
                    'equipment_name':names[row['equipment_id']], 'class':row['reviewed_class'] or row['class_name'],
                    'coordinate_system':'image_relative','unit':'pixel','model_id':row['model_id'],
                    'bbox_xyxy':[row[k] for k in ('bbox_x1','bbox_y1','bbox_x2','bbox_y2')],
                    'cv_confidence':row['confidence'],'review_status':row['review_status'],
                    'model_provenance':payload.get('model_provenance'),
                    'association_review':review,'usable_as_identity_evidence':verified})
        diagnosis = state.get('diagnosis')
        findings = {f.finding_id:f for f in diagnosis.findings if f.equipment_id in ids} if diagnosis else {}
        operations = []
        for item in state.get('opportunities',[]):
            supported = [findings[k] for k in item.related_finding_ids if k in findings and findings[k].equipment_id==item.equipment_id and findings[k].evidence]
            if not supported: continue
            operations.append({'opportunity_id':item.opportunity_id,'equipment_id':item.equipment_id,
                'equipment_name':names.get(item.equipment_id),'finding_ids':[f.finding_id for f in supported],
                'finding_type':supported[0].finding_type,'origin':'bems_deterministic_diagnosis',
                'evidence':[e.to_dict() for f in supported for e in f.evidence],
                'drawing_identity_verified':item.equipment_id in verified_ids,
                'estimated_savings':None})
        unavailable = [{'equipment_id':k.equipment_id,'equipment_name':k.equipment_name,'reason':k.reason}
                       for k in (diagnosis.analytics.equipment_kpis if diagnosis else [])
                       if k.equipment_id in ids and k.status!='available']
        classes = dict(Counter(r['reviewed_class'] or r['class_name'] for r in confirmed))
        linked_ops = any(x['drawing_identity_verified'] for x in operations)
        return {'schema_version':1,'project_id':project.project_id,'equipment_filter':equipment_name,
            'equipment_found':bool(selected) if equipment_name else None,'data_revision':project.data_revision,
            'workflow_status':'evidence_linked' if linked_ops else 'partial',
            'stages':{'drawing_objects':len(confirmed),'bems_available':bool(state.get('data_available')),
                      'associated_objects':len(links),'verified_associations':sum(x['usable_as_identity_evidence'] for x in links),
                      'diagnosis_completed':diagnosis is not None,'supported_operating_actions':len(operations)},
            'drawing_evidence':{'predicted_count':sum(r['review_status']=='predicted' for r in detections),
                                'confirmed_classes':classes,'associations':links,'invalid_drawings':invalid_drawings},
            'bems_evidence':{'imports':[{'file_hash':i.get('file_hash')} for i in state.get('metadata',{}).get('imports',[])],
                             'unavailable_equipment':unavailable},
            'layers':{'operation':{'status':'supported' if operations else 'insufficient_evidence','items':operations},
                      'equipment_retrofit':{'status':'assessment_required','origin':'general_review_checklist','checklist_key':'retrofit_equipment_checklist'},
                      'building_retrofit':{'status':'assessment_required','origin':'general_review_checklist','checklist_key':'retrofit_building_checklist'}},
            'design_ready':False,'estimated_savings':None,
            'boundaries':['no_pixel_to_building_transform','no_automatic_equipment_identity','no_inferred_topology',
                          'no_envelope_properties_from_boxes','no_savings_or_investment_estimate']}

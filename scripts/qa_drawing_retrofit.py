"""Native GUI review of local drawings/BEMS; candidate links prove no identity.

Use qa_functional_desktop.py with a private manifest containing retrofit_review,
weights, drawings and candidate_equipment. Outputs remain in its local workspace.
No source asset or absolute path is bundled with this runner.
"""
import json
import pickle
from dataclasses import asdict
from PyQt5.QtCore import Qt
from building_ai.i18n import LanguageManager
from scripts.qa_functional_workflows import exercise_ai


def exercise_retrofit(window, context, manifest, workspace, report, navigate,
                      click, events, answers, save, reopen=False):
    assert context.dataframe is not None
    pid = context.current_project.project_id
    navigate('drawing_intelligence'); page = window.page_by_key['drawing_intelligence']
    if reopen:
        previous = json.loads((workspace/'retrofit-evidence.json').read_text(encoding='utf-8'))
        restored = context.agent.get_retrofit_review(pid,previous['equipment_filter'])
        assert restored == previous, 'Persisted review changed across restart'
        click(page.retrofit_button); dialog = page.retrofit_dialog
        dialog.grab().save(str(workspace/'retrofit-reopened.png')); click(dialog.close_button)
        report['retrofit_reopened'] = True; save(); return
    answers.append(manifest['weights']); click(page.model_button)
    report['retrofit_drawings'] = []
    for index, source in enumerate(manifest['drawings']):
        answers.append(source); click(page.import_button); assert page.pix
        click(page.detect_button); assert page.rows, 'No actual YOLO predictions'
        candidates = [(i,r) for i,r in enumerate(page.rows) if r['class_name']=='aircon'
                      and r['bbox_x1']>1 and r['bbox_x2']<r['image_width']-1]
        selected, prediction = max(candidates,key=lambda pair:pair[1]['confidence'])
        page.list.setCurrentRow(selected); page.class_choice.setCurrentText('aircon')
        click(page.confirm)
        assert page.selected()['review_status']=='confirmed'
        equipment = next(e for e in context.equipment if e.name==manifest['candidate_equipment'])
        page.map_equipment.setCurrentIndex(page.map_equipment.findData(equipment.equipment_id))
        page.association_status.setCurrentIndex(page.association_status.findData('pending'))
        page.association_reviewer.setText('LOCAL SOFTWARE QA')
        page.association_note.setText('Candidate only: sources lack evidence of the same building and equipment. Visual symbol review is not device identity verification.')
        click(page.map_button)
        confirmed_id = prediction['detection_id']
        positions=context.drawings.image_positions(pid,page.drawing['drawing_id'])
        assert len(positions)==1 and positions[0]['association_review']['association_status']=='pending'
        assert positions[0]['model_provenance']['weights_sha256']
        answers.append(str(workspace/f'retrofit-positions-{index}.json')); click(page.export_button)
        click(page.zoom_in); click(page.zoom_out); click(page.fit)
        assert positions==context.drawings.image_positions(pid,page.drawing['drawing_id'])
        click(page.open_equipment)
        assert window.stack.currentWidget() is window.page_by_key['equipment']
        window.grab().save(str(workspace/f'retrofit-equipment-{index}.png'))
        navigate('energy_analysis'); window.grab().save(str(workspace/f'retrofit-energy-{index}.png'))
        navigate('analysis'); window.grab().save(str(workspace/f'retrofit-diagnosis-{index}.png'))
        navigate('drawing_intelligence')
        window.grab().save(str(workspace/f'retrofit-drawing-{index}.png'))
        report['retrofit_drawings'].append({'drawing_id':page.drawing['drawing_id'],
            'predictions':len(page.rows),'classes':sorted({r['class_name'] for r in page.rows}),
            'reviewed_object':confirmed_id,'association':'pending','equipment_candidate':equipment.name,
            'model_provenance':positions[0]['model_provenance']})
        save()
    click(page.retrofit_button); dialog=page.retrofit_dialog
    assert dialog.report['workflow_status']=='partial' and not dialog.report['stages']['verified_associations']
    answers.append(str(workspace/'retrofit-evidence.json')); click(dialog.export_button)
    assert json.loads((workspace/'retrofit-evidence.json').read_text(encoding='utf-8'))==dialog.report
    # Native visible dialog receives real language-change signals; only rendering changes.
    original=pickle.dumps(asdict(context.diagnosis_result))
    for locale in ('zh_CN','en_US','ja_JP'):
        LanguageManager.instance().set_language(locale); events()
        assert 'retrofit_' not in dialog.body.toPlainText()
        dialog.grab().save(str(workspace/f'retrofit-{locale}.png'))
    assert pickle.dumps(asdict(context.diagnosis_result))==original
    click(dialog.close_button)
    # Also reach the same entry through Diagnostics, using its actual button.
    navigate('analysis'); analysis=window.page_by_key['analysis']; click(analysis.retrofit_button)
    assert analysis.retrofit_dialog.report['workflow_status']=='partial'
    click(analysis.retrofit_dialog.close_button)
    exercise_ai(window,context,manifest,workspace,report,navigate,click,events,save)
    assert any(call['tool']=='get_retrofit_review' for trace in report['traces'] for call in trace['tool_calls'])
    report['retrofit_review']={'native_platform':'windows','workflow_status':'partial',
        'identity_verified':False,'languages':['zh_CN','en_US','ja_JP'],'evidence_export':'PASS',
        'scientific_results_unchanged':pickle.dumps(asdict(context.diagnosis_result))==original}
    save()

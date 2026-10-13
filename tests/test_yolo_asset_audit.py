from scripts.audit_yolo_dataset import audit
from PIL import Image


def test_audit_detects_bad_boxes_and_split_leakage_without_changing_sources(tmp_path):
    for split in ('train','test'):
        (tmp_path/split/'images').mkdir(parents=True)
        (tmp_path/split/'labels').mkdir()
        Image.new('RGB',(20,20)).save(tmp_path/split/'images'/'sample.rf.variant.png')
    good=tmp_path/'train/labels/sample.rf.variant.txt'
    good.write_text('0 .5 .5 .2 .2\n',encoding='utf-8')
    bad=tmp_path/'test/labels/sample.rf.variant.txt'
    bad.write_text('0 .01 .5 .2 .2\n3 .5 .5 .2 .2\n0 nan .5 .2 .2\n',encoding='utf-8')
    before={str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    report=audit(tmp_path)
    assert len(report['invalid_labels'])==3
    assert len(report['exact_cross_split_duplicates'])==1
    assert report['source_name_cross_split_candidates']=={'sample':['test','train']}
    assert report['splits']['train']['boxes_by_class']=={0:1}
    assert {str(p):p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}==before

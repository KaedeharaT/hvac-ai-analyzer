"""Synthetic geometry; no private historical model is distributed."""
import hashlib
import json
from pathlib import Path

import pytest

from building_ai.services.model_viewer_service import ModelViewerService
from building_ai.storage import Database, ProjectStore
from building_ai.vision.model_geometry import read_osm


def synthetic_osm(rotation="", x_origin="", coordinates="0,0,0, 4,0,0, 4,3,0, 0,3,0", version="3.7.0"):
    return (f"OS:Version,version,{version};\n"
        f"OS:Space,space,Synthetic space,,,,{rotation},{x_origin},,,;\n"
        f"OS:Surface,surface,Synthetic floor,Floor,,space,Ground,,NoSun,NoWind,,4,{coordinates};\n").encode()


def test_source_geometry_units_and_hash_are_preserved():
    data = synthetic_osm()
    model = read_osm(data)
    assert model["unit"] == "m" and model["coordinate_system"] == "model_local"
    assert model["bounds"] == [[0,0,0],[4,3,0]]
    assert model["elements"][0]["vertices"] == [[0,0,0],[4,0,0],[4,3,0],[0,3,0]]
    assert model["model_hash"] == hashlib.sha256(data).hexdigest()
    assert model["engineering_validation"] == "not_verified_as_built"


@pytest.mark.parametrize("options", [
    {"rotation":"45"}, {"x_origin":"12"}, {"version":"4.0.0"},
    {"coordinates":"0,0,0, 4,0,0, 4,3,1, 0,3,0"},
    {"coordinates":"0,0,0, 4,0,0, 4,3,nan, 0,3,0"},
    {"coordinates":"0,0,0, 0,0,0, 4,3,0, 0,3,0"},
])
def test_unsupported_or_invalid_geometry_fails_explicitly(options):
    with pytest.raises(ValueError): read_osm(synthetic_osm(**options))


def test_source_subsurface_reference_and_counts():
    base = synthetic_osm()
    opening = b"OS:SubSurface,window,Window,FixedWindow,,surface,,,,1,4,1,1,0,2,1,0,2,2,0,1,2,0;"
    model = read_osm(base + opening)
    assert len(model["elements"]) == 2 and model["elements"][1]["parent_id"] == "surface"
    with pytest.raises(ValueError, match="space"):
        read_osm(base + opening.replace(b",surface,",b",missing,"))
    with pytest.raises(ValueError, match="vertex count"):
        read_osm(base.replace(b",4,0,0,0",b",5,0,0,0"))
    with pytest.raises(ValueError, match="separate geometric"):
        read_osm(base + opening.replace(b",1,4,",b",2,4,"))


@pytest.fixture
def models(tmp_path):
    database = Database(tmp_path/"data.sqlite")
    projects = ProjectStore(database); a=projects.create("A"); b=projects.create("B")
    from building_ai.models import SemanticResult
    from building_ai.services.equipment_service import EquipmentService
    identifiers=[]
    for project in (a,b):
        points=[SemanticResult("Synthetic power", "heat_source_power", point_id="p-"+project.project_id, equipment_id="AHP-01",unit="kW",physical_quantity="power")]
        projects.save_semantics(project.project_id,points)
        identifiers.append(EquipmentService().organize(project.project_id,points).equipment[0].equipment_id)
    source = tmp_path/"synthetic.osm"; source.write_bytes(synthetic_osm())
    return ModelViewerService(database,tmp_path/"models"),a,b,source,*identifiers


def test_project_scoping_explicit_association_and_export(models,tmp_path):
    service,a,b,source,equipment_a,equipment_b=models
    model=service.import_model(a.project_id,source); mid=model["model_id"]
    assert service.list_models(b.project_id)==[]
    assert service.associations(a.project_id,mid)==[]
    with pytest.raises(KeyError): service.load(b.project_id,mid)
    with pytest.raises(KeyError): service.load(a.project_id,"../escape")
    with pytest.raises(KeyError): service.list_models("../escape")
    with pytest.raises(ValueError): service.associate(a.project_id,mid,"surface",equipment_b,"QA","Synthetic association")
    with pytest.raises(ValueError): service.associate(a.project_id,mid,"surface",equipment_a,"","Missing reviewer")
    event=service.associate(a.project_id,mid,"surface",equipment_a,"Synthetic QA","Explicit reference, not installation position")
    assert event["drawing_hash"] is None
    assert event["review_status"]=="human_confirmed" and event["association_kind"]=="reference_only_not_equipment_position"
    service.import_model(a.project_id,source)  # Reimport must retain the review history.
    assert service.associations(a.project_id,mid)==[event]
    reopened=ModelViewerService(service.database,service.root)
    assert reopened.load(a.project_id,mid)["model_hash"]==mid
    destination=tmp_path/"provenance.json"; reopened.export_provenance(a.project_id,mid,destination)
    payload=json.loads(destination.read_text(encoding="utf-8"))
    assert payload["association_events"]==[event]
    assert payload["coordinate_system"]=="model_local" and payload["unit"]=="m"
    assert str(tmp_path) not in destination.read_text(encoding="utf-8")
    assert source.read_bytes()==synthetic_osm()


def test_tampered_model_and_unsupported_formats(models):
    service,a,b,source,equipment_a,equipment_b=models
    model=service.import_model(a.project_id,source); mid=model["model_id"]
    with pytest.raises(ValueError): service.import_model(a.project_id,source.with_suffix(".rvt"))
    snapshot=service.root/a.project_id/"models"/mid/"source.osm"
    snapshot.write_bytes(synthetic_osm()+b"! changed")
    with pytest.raises(ValueError,match="changed"): service.load(a.project_id,mid)


def test_optional_viewer_is_not_imported_with_main_window():
    # Importing the UI must not require an optional renderer or a Revit installation.
    import subprocess,sys
    code="import sys; from building_ai.ui.main_window import MainWindow; assert 'PyQt5.QtWebEngineWidgets' not in sys.modules; assert not any('Revit' in m for m in sys.modules)"
    result=subprocess.run([sys.executable,"-c",code],capture_output=True,text=True)
    assert result.returncode==0,result.stderr


def test_application_and_model_dialog_work_without_optional_webengine(tmp_path):
    import os
    import subprocess
    import sys
    code = """
import importlib.abc
import sys
from pathlib import Path
class WithoutWebEngine(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith('PyQt5.QtWebEngine'):
            raise ModuleNotFoundError(fullname)
sys.meta_path.insert(0, WithoutWebEngine())
from building_ai.ui.model_viewer_backend import prepare_model_viewer
assert prepare_model_viewer() is False
from PyQt5.QtWidgets import QApplication
from building_ai.config import Settings
from building_ai.ui.context import ApplicationContext
from building_ai.ui.main_window import MainWindow
from building_ai.i18n import tr
app = QApplication([])
context = ApplicationContext(Settings(data_dir=Path(sys.argv[1]) / 'runtime'))
context.current_project = context.projects.create('Synthetic optional dependency check')
window = MainWindow(context)
window.show()
window.navigate_to('drawing_intelligence')
page = window.page_by_key['drawing_intelligence']
page.preview_button.click()
app.processEvents()
viewer = page.model_dialog
assert viewer.view is None and viewer.model is None
assert viewer.status.text() == tr('model_optional')
viewer.close()
window.navigate_to('energy_analysis')
app.processEvents()
window.close()
"""
    environment = {**os.environ, "QT_QPA_PLATFORM": "offscreen",
                   "BUILDING_AI_CONFIG_DIR": str(tmp_path / "config")}
    result = subprocess.run([sys.executable, "-c", code, str(tmp_path)], env=environment,
                            capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stderr

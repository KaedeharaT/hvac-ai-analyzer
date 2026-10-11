"""Optional, offline preview under Drawing Intelligence, with project evidence."""
from __future__ import annotations

import json
from pathlib import Path

from PyQt5.QtCore import QObject, Qt, QUrl, pyqtSlot
from PyQt5.QtWidgets import (QComboBox, QDialog, QFileDialog, QFormLayout, QHBoxLayout,
    QLabel, QLineEdit, QListWidget, QListWidgetItem, QMessageBox, QPushButton,
    QScrollArea, QSplitter, QTextEdit, QVBoxLayout, QWidget)

from building_ai.i18n import LanguageManager, tr
from building_ai.services.model_viewer_service import ModelViewerService
from building_ai.ui.analysis_renderer import finding_text

ASSETS = Path(__file__).resolve().parents[1] / "model_viewer_assets"


class Bridge(QObject):
    def __init__(self, dialog):
        super().__init__(dialog); self.dialog = dialog

    @pyqtSlot()
    def ready(self):
        self.dialog.web_ready = True
        self.dialog.render_model()

    @pyqtSlot(str)
    def selected(self, object_id):
        for row in range(self.dialog.objects.count()):
            if self.dialog.objects.item(row).data(Qt.UserRole) == object_id:
                self.dialog.objects.setCurrentRow(row); break

    @pyqtSlot(str)
    def failed(self, detail):
        self.dialog.web_error = detail
        self.dialog.status.setText(tr("model_error", detail=detail))


class ModelViewerDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.context = context
        self.project_id = context.current_project.project_id
        self.service = ModelViewerService(context.database, context.settings.timeseries_dir)
        self.model = None; self.web_ready = False; self.web_error = None; self.view = None
        self.resize(1180, 760); self.setModal(True)
        layout = QVBoxLayout(self)
        self.boundary = QLabel(); self.boundary.setWordWrap(True); layout.addWidget(self.boundary)
        bar = QHBoxLayout(); self.models = QComboBox(); bar.addWidget(self.models, 1)
        self.import_button = QPushButton(); self.reset_button = QPushButton(); self.export_button = QPushButton()
        for button in (self.import_button, self.reset_button, self.export_button): bar.addWidget(button)
        layout.addLayout(bar)
        self.status = QLabel(); self.status.setWordWrap(True); layout.addWidget(self.status)
        self.controls = QLabel(); layout.addWidget(self.controls)
        self.splitter = QSplitter(); layout.addWidget(self.splitter, 1)
        self.canvas_container = QWidget(); self.canvas_layout = QVBoxLayout(self.canvas_container)
        self.canvas_layout.setContentsMargins(0, 0, 0, 0); self.splitter.addWidget(self.canvas_container)
        side = QWidget(); box = QVBoxLayout(side)
        self.category_title = QLabel(); box.addWidget(self.category_title)
        self.categories = QListWidget(); self.categories.setMaximumHeight(140); box.addWidget(self.categories)
        self.object_title = QLabel(); box.addWidget(self.object_title)
        self.objects = QListWidget(); self.objects.setMinimumHeight(110); box.addWidget(self.objects)
        self.details = QTextEdit(); self.details.setReadOnly(True); self.details.setMinimumHeight(170); box.addWidget(self.details)
        self.reference_note = QLabel(); self.reference_note.setWordWrap(True); box.addWidget(self.reference_note)
        form = QFormLayout(); self.equipment = QComboBox(); self.reviewer = QLineEdit(); self.note = QLineEdit()
        self.equipment_label = QLabel(); self.reviewer_label = QLabel(); self.note_label = QLabel()
        form.addRow(self.equipment_label, self.equipment); form.addRow(self.reviewer_label, self.reviewer); form.addRow(self.note_label, self.note); box.addLayout(form)
        self.confirm_button = QPushButton(); self.open_equipment = QPushButton()
        box.addWidget(self.confirm_button); box.addWidget(self.open_equipment)
        self.side_scroll = QScrollArea(); self.side_scroll.setWidgetResizable(True); self.side_scroll.setWidget(side)
        self.side_scroll.setMinimumWidth(310); self.splitter.addWidget(self.side_scroll); self.splitter.setSizes([780,380])
        self.import_button.clicked.connect(self.import_model)
        self.models.currentIndexChanged.connect(self.load_selected_model)
        self.reset_button.clicked.connect(lambda: self.js("window.viewer.reset()"))
        self.categories.itemChanged.connect(self.filter_category)
        self.objects.currentRowChanged.connect(self.show_object)
        self.confirm_button.clicked.connect(self.confirm_association)
        self.open_equipment.clicked.connect(self.navigate_equipment)
        self.export_button.clicked.connect(self.export_provenance)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.initialize_web_view()
        self.retranslate_ui(); self.refresh_models()

    def initialize_web_view(self):
        try:
            from PyQt5.QtWebChannel import QWebChannel
            from PyQt5.QtWebEngineWidgets import QWebEngineView, QWebEnginePage, QWebEngineProfile, QWebEngineSettings
            from PyQt5.QtWebEngineCore import QWebEngineUrlRequestInterceptor
        except (ImportError, RuntimeError):
            self.missing = QLabel(); self.missing.setWordWrap(True); self.canvas_layout.addWidget(self.missing)
            return
        class OfflineRequests(QWebEngineUrlRequestInterceptor):
            def interceptRequest(self, info):
                url = info.requestUrl()
                allowed = url.scheme() == "qrc" and url.path() == "/qtwebchannel/qwebchannel.js"
                if url.isLocalFile():
                    allowed = Path(url.toLocalFile()).resolve().parent == ASSETS.resolve()
                info.block(not allowed)
        class OfflinePage(QWebEnginePage):
            def acceptNavigationRequest(self, url, kind, main_frame):
                return url.isLocalFile() and Path(url.toLocalFile()).resolve() == (ASSETS / "index.html").resolve()
        self.profile = QWebEngineProfile(self)
        self.interceptor = OfflineRequests(self.profile); self.profile.setUrlRequestInterceptor(self.interceptor)
        self.view = QWebEngineView(); self.page = OfflinePage(self.profile, self.view); self.view.setPage(self.page)
        self.page.settings().setAttribute(QWebEngineSettings.LocalContentCanAccessRemoteUrls, False)
        self.page.settings().setAttribute(QWebEngineSettings.JavascriptCanOpenWindows, False)
        self.channel = QWebChannel(self.page); self.bridge = Bridge(self)
        self.channel.registerObject("bridge", self.bridge); self.page.setWebChannel(self.channel)
        self.canvas_layout.addWidget(self.view); self.view.setUrl(QUrl.fromLocalFile(str(ASSETS / "index.html")))

    def js(self, source):
        if self.view and self.web_ready: self.page.runJavaScript(source)

    def check_project(self):
        if not self.context.current_project or self.context.current_project.project_id != self.project_id:
            raise ValueError("The active project changed; reopen the model viewer")

    def error(self, exc):
        message = str(exc)
        if "Reviewer and association evidence" in message:
            key = "model_review_required"
        elif "Nonidentity space" in message:
            key = "model_transform_unsupported"
        elif "supports OSM" in message or "Select an OSM" in message:
            key = "model_format_unsupported"
        elif "snapshot changed" in message or "provenance mismatch" in message:
            key = "model_source_changed"
        elif isinstance(exc, KeyError) or "project" in message.lower():
            key = "model_project_mismatch"
        else:
            key = "model_invalid_geometry"
        dialog = QMessageBox(QMessageBox.Warning, tr("model_viewer"), tr(key), parent=self)
        # Preserve the original technical diagnostic separately from the localized guidance.
        dialog.setDetailedText(message)
        dialog.exec_()

    def refresh_models(self, model_id=None):
        self.models.blockSignals(True); self.models.clear()
        for item in self.service.list_models(self.project_id): self.models.addItem(item["source_name"], item["model_id"])
        if model_id: self.models.setCurrentIndex(max(0, self.models.findData(model_id)))
        self.models.blockSignals(False); self.load_selected_model()

    def import_model(self):
        path, _ = QFileDialog.getOpenFileName(self, tr("model_import"), "", "OpenStudio (*.osm)")
        if path: self.import_path(path)

    def import_path(self, path):
        try:
            self.check_project(); model = self.service.import_model(self.project_id, path)
            self.refresh_models(model["model_id"])
        except (OSError, ValueError, KeyError) as exc: self.error(exc)

    def load_selected_model(self):
        model_id = self.models.currentData()
        if not model_id: self.model = None; self.retranslate_ui(); return
        try:
            self.check_project(); self.model = self.service.load(self.project_id, model_id)
            self.events = self.service.associations(self.project_id, model_id)
        except (OSError, ValueError, KeyError) as exc: self.error(exc); return
        self.objects.blockSignals(True); self.objects.clear()
        for element in self.model["elements"]:
            item = QListWidgetItem(element["name"]); item.setData(Qt.UserRole, element["object_id"]); self.objects.addItem(item)
        self.objects.blockSignals(False)
        self.categories.blockSignals(True); self.categories.clear()
        for category, count in self.model["categories"].items():
            item = QListWidgetItem(); item.setData(Qt.UserRole, category); item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked); self.categories.addItem(item)
        self.categories.blockSignals(False)
        self.retranslate_ui(); self.render_model()

    def render_model(self):
        if self.web_ready and self.model:
            self.js("window.viewer.load(" + json.dumps(self.model, ensure_ascii=True, allow_nan=False) + ")")
        self.update_status()

    def update_status(self):
        self.reset_button.setEnabled(self.web_ready and bool(self.model)); self.export_button.setEnabled(bool(self.model))
        if self.web_error: self.status.setText(tr("model_error", detail=self.web_error))
        elif not self.view: self.status.setText(tr("model_optional"))
        elif not self.web_ready: self.status.setText(tr("model_loading"))
        elif not self.model: self.status.setText(tr("model_empty"))
        else: self.status.setText(tr("model_loaded", count=len(self.model["elements"]), system=self.model["coordinate_system"], unit=self.model["unit"]))

    def retranslate_ui(self):
        self.setWindowTitle(tr("model_viewer"))
        for widget, key in ((self.import_button,"model_import"),(self.reset_button,"model_reset"),
            (self.export_button,"model_export"),(self.boundary,"model_boundary"),(self.controls,"model_controls"),
            (self.category_title,"model_categories"),(self.object_title,"model_objects"),
            (self.reference_note,"model_reference_only"),(self.equipment_label,"equipment"),
            (self.reviewer_label,"model_reviewer"),(self.note_label,"model_note"),
            (self.confirm_button,"model_confirm"),(self.open_equipment,"drawing_open_equipment")):
            widget.setText(tr(key))
        if not self.view: self.missing.setText(tr("model_optional"))
        chosen = self.equipment.currentData(); self.equipment.blockSignals(True); self.equipment.clear()
        self.equipment.addItem(tr("drawing_choose_equipment"), None)
        for item in self.context.equipment: self.equipment.addItem(item.name, item.equipment_id)
        self.equipment.setCurrentIndex(max(0, self.equipment.findData(chosen))); self.equipment.blockSignals(False)
        self.categories.blockSignals(True)
        for row in range(self.categories.count()):
            item = self.categories.item(row); category = item.data(Qt.UserRole)
            label = tr("model_type_"+category.lower())
            if label.startswith("model_type_"): label = category
            item.setText(f"{label} ({self.model['categories'][category]})")
        self.categories.blockSignals(False); self.update_status(); self.show_object()

    def filter_category(self, item):
        self.js("window.viewer.showCategory("+json.dumps(item.data(Qt.UserRole))+","+json.dumps(item.checkState()==Qt.Checked)+")")

    def selected_object(self):
        item = self.objects.currentItem()
        return next((e for e in self.model["elements"] if e["object_id"]==item.data(Qt.UserRole)), None) if self.model and item else None

    def current_association(self):
        obj = self.selected_object()
        return next((e for e in reversed(getattr(self,"events",[])) if obj and e["object_id"]==obj["object_id"]), None)

    def show_object(self, *_):
        obj = self.selected_object(); association = self.current_association()
        self.confirm_button.setEnabled(bool(obj)); self.open_equipment.setEnabled(bool(association))
        if not obj: self.details.setPlainText(tr("model_surface_note")); return
        self.js("window.viewer.select("+json.dumps(obj["object_id"])+")")
        category = tr("model_type_" + obj["category"].lower())
        if category.startswith("model_type_"): category = obj["category"]
        lines = [obj["name"], obj["object_id"], category, obj["space_name"], tr("model_origin"),
            tr("model_hash")+": "+self.model["model_hash"],tr("model_surface_note")]
        if self.model["omitted_geometry"]: lines.append(tr("model_omitted",count=sum(self.model["omitted_geometry"].values())))
        if not association: lines.append(tr("model_no_link"))
        else:
            equipment_id = association["equipment_id"]
            equipment = next((e for e in self.context.equipment if e.equipment_id == equipment_id), None)
            if equipment is None: lines.append(tr("model_no_link"))
            else:
                lines += [tr("equipment")+": "+equipment.name, tr("model_reviewer")+": "+association["reviewer"],
                    association["confirmed_at"], association["note"]]
                points = self.context.semantic_result.semantic_results if self.context.semantic_result else []
                signals = [p.raw_name for p in points if p.effective_equipment_id == equipment.name]
                lines.append(tr("equipment_signals")+": "+", ".join(signals))
                diagnosis = self.context.diagnosis_result
                kpi = next((k for k in diagnosis.analytics.equipment_kpis if k.equipment_id==equipment_id),None) if diagnosis else None
                for key, label in (("cop",tr("analysis_metric_mean_cop")),
                                   ("delta_t_c",tr("analysis_metric_mean_delta_t_c"))):
                    value = kpi.metric_summary.get(key,{}).get("mean") if kpi else None
                    lines.append(label+": "+(f"{value:.3g}" if value is not None else "—"))
                lines += [finding_text(f)[0] for f in diagnosis.findings if f.equipment_id==equipment_id] if diagnosis else []
                locations = self.context.drawings.equipment_location(self.project_id, equipment_id)
                lines.append(tr("equipment_drawing")+": "+(", ".join(sorted({x['file_name'] for x in locations})) or "—"))
        self.details.setPlainText("\n\n".join(lines))

    def confirm_association(self):
        obj = self.selected_object()
        if not obj or not self.equipment.currentData(): return
        try:
            self.check_project(); self.service.associate(self.project_id,self.model["model_id"],obj["object_id"],
                self.equipment.currentData(),self.reviewer.text(),self.note.text())
            self.events=self.service.associations(self.project_id,self.model["model_id"]); self.show_object()
        except (OSError,ValueError,KeyError) as exc: self.error(exc)

    def navigate_equipment(self):
        association=self.current_association()
        if not association: return
        self.check_project()
        equipment=next((e for e in self.context.equipment if e.equipment_id==association["equipment_id"]),None)
        if equipment and self.parent():
            self.parent().navigation_requested.emit("equipment",{"equipment_id":equipment.name}); self.accept()

    def export_provenance(self):
        if not self.model: return
        path,_=QFileDialog.getSaveFileName(self,tr("model_export"),"model-provenance.json","JSON (*.json)")
        if not path: return
        try: self.check_project(); self.service.export_provenance(self.project_id,self.model["model_id"],path)
        except (OSError,ValueError,KeyError) as exc: self.error(exc); return
        self.status.setText(tr("model_saved"))

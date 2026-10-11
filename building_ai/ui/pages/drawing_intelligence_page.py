from __future__ import annotations
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtWidgets import QComboBox, QFileDialog, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QScrollArea, QFrame, QSplitter, QVBoxLayout, QWidget
from building_ai.i18n import LanguageManager, tr
from building_ai.vision import DrawingModelRegistry, YOLODrawingDetector

class DrawingView(QGraphicsView):
    object_clicked = pyqtSignal(str)

    def mousePressEvent(self, event):
        item = self.itemAt(event.pos())
        if item is not None and item.data(0):
            self.object_clicked.emit(str(item.data(0)))
        super().mousePressEvent(event)


class DrawingIntelligencePage(QWidget):
    navigation_requested = pyqtSignal(str, object)
    COLORS={"aircon":"#2563EB","window":"#059669","baseline":"#64748B","baseline_mark":"#64748B"}
    def __init__(self, context):
        super().__init__(); self.context=context; self.drawing=None; self.rows=[]; self.pix=None
        root=QVBoxLayout(self); root.setContentsMargins(24,24,24,24)
        self.title=QLabel(); self.title.setObjectName("PageTitle"); root.addWidget(self.title); self.subtitle=QLabel(); self.subtitle.setObjectName("Muted"); self.subtitle.setWordWrap(True); root.addWidget(self.subtitle)
        self.boundary=QLabel(); self.boundary.setWordWrap(True); root.addWidget(self.boundary)
        selector=QHBoxLayout(); self.drawing_label=QLabel(); self.drawing_choice=QComboBox(); self.drawing_choice.currentIndexChanged.connect(self.change_drawing); selector.addWidget(self.drawing_label); selector.addWidget(self.drawing_choice,1); self.export_button=QPushButton(); self.export_button.clicked.connect(self.export_positions); selector.addWidget(self.export_button); root.addLayout(selector)
        bar=QHBoxLayout(); self.import_button=QPushButton(); self.model_button=QPushButton(); self.detect_button=QPushButton(); self.zoom_out=QPushButton("−"); self.fit=QPushButton("Fit"); self.zoom_in=QPushButton("+"); self.model_status=QLabel(); self.model_status.setObjectName("StatusBadge")
        for x in (self.import_button,self.model_button,self.detect_button,self.zoom_out,self.fit,self.zoom_in,self.model_status): bar.addWidget(x)
        bar.addStretch(1); root.addLayout(bar); split=QSplitter(); self.scene=QGraphicsScene(self); self.viewer=DrawingView(self.scene); self.viewer.object_clicked.connect(self.select_object); self.viewer.setDragMode(QGraphicsView.ScrollHandDrag); split.addWidget(self.viewer)
        side=QWidget(); sidebox=QVBoxLayout(side); self.summary=QLabel(); self.summary.setWordWrap(True); self.list=QListWidget(); self.list.setMinimumHeight(120); self.class_choice=QComboBox(); self.class_choice.addItems(["aircon","baseline_mark","window"]); self.confirm=QPushButton(); self.reject=QPushButton(); self.map_equipment=QComboBox(); self.map_button=QPushButton(); self.open_equipment=QPushButton(); self.open_equipment.setObjectName("PrimaryButton"); self.open_equipment.clicked.connect(self.open_selected_equipment)
        self.review_label=QLabel(); self.mapping_label=QLabel(); self.evidence_note=QLabel(); self.evidence_note.setObjectName("Muted"); self.evidence_note.setWordWrap(True)
        for x in (self.summary,self.list,self.review_label,self.class_choice,self.confirm,self.reject,self.mapping_label,self.map_equipment,self.map_button,self.open_equipment,self.evidence_note): sidebox.addWidget(x)
        self.review_scroll=QScrollArea(); self.review_scroll.setWidgetResizable(True); self.review_scroll.setFrameShape(QFrame.NoFrame); self.review_scroll.setMinimumWidth(280); self.review_scroll.setWidget(side); split.addWidget(self.review_scroll); split.setSizes([850,330]); root.addWidget(split,1)
        self.import_button.clicked.connect(self.import_drawing); self.model_button.clicked.connect(self.choose_model); self.detect_button.clicked.connect(self.detect); self.zoom_in.clicked.connect(lambda:self.viewer.scale(1.2,1.2)); self.zoom_out.clicked.connect(lambda:self.viewer.scale(.8,.8)); self.fit.clicked.connect(self.fit_view); self.list.currentRowChanged.connect(self.highlight); self.confirm.clicked.connect(lambda:self.review("confirmed")); self.reject.clicked.connect(lambda:self.review("rejected")); self.map_button.clicked.connect(self.map_selected); LanguageManager.instance().language_changed.connect(self.retranslate_ui); self.retranslate_ui()
    def retranslate_ui(self):
        for widget, key in ((self.title, "drawing_title"), (self.subtitle, "drawing_subtitle"),
                (self.boundary, "drawing_boundary"), (self.import_button, "drawing_import"),
                (self.model_button, "drawing_model"), (self.detect_button, "drawing_detect"),
                (self.confirm, "drawing_confirm"), (self.reject, "drawing_reject"),
                (self.map_button, "drawing_map"), (self.review_label, "drawing_review_class"),
                (self.mapping_label, "drawing_association"), (self.open_equipment, "drawing_open_equipment"),
                (self.evidence_note, "drawing_evidence"), (self.fit, "drawing_fit"),
                (self.export_button, "drawing_export"), (self.drawing_label, "drawing_select")):
            widget.setText(tr(key))
        self.zoom_in.setToolTip(tr("drawing_zoom_in")); self.zoom_out.setToolTip(tr("drawing_zoom_out"))
        self.refresh()
        if self.drawing: self.refresh_detections()

    def refresh(self):
        project=self.context.current_project
        drawings=self.context.drawings.list_drawings(project.project_id) if project else []
        if self.drawing and self.drawing['drawing_id'] not in {d['drawing_id'] for d in drawings}:
            self.drawing=None; self.rows=[]; self.pix=None; self.scene.clear(); self.list.clear(); self.summary.clear(); self.map_equipment.clear(); self.evidence_note.setText(tr("drawing_evidence"))
        self.drawing_choice.blockSignals(True); self.drawing_choice.clear()
        for drawing in drawings: self.drawing_choice.addItem(drawing['file_name'],drawing)
        if drawings and self.drawing is None: self.select_drawing(drawings[0])
        if self.drawing:
            self.drawing_choice.setCurrentIndex(next((i for i,d in enumerate(drawings) if d['drawing_id']==self.drawing['drawing_id']),0))
        self.drawing_choice.blockSignals(False)
        registry=DrawingModelRegistry(self.context.settings.drawing_model_path)
        self.model_status.setText(tr("no_project") if not project else tr("drawing_model_ready" if registry.configured() else "drawing_model_missing"))
        self.export_button.setEnabled(bool(self.drawing)); self.detect_button.setEnabled(bool(self.drawing))

    def change_drawing(self):
        drawing=self.drawing_choice.currentData()
        if drawing: self.select_drawing(drawing)

    def select_object(self, object_id):
        index=next((i for i,r in enumerate(self.rows) if r['detection_id']==object_id),-1)
        self.list.setCurrentRow(index)

    def export_positions(self):
        if not self.drawing: return
        path,_=QFileDialog.getSaveFileName(self,tr("drawing_export"),"image-positions.json","JSON (*.json)")
        if not path: return
        try: self.context.drawings.export_image_positions(self.context.current_project.project_id,self.drawing['drawing_id'],path)
        except (ValueError,OSError,KeyError) as exc:
            QMessageBox.warning(self,tr("drawing_error"),tr("drawing_failure",detail=str(exc))); return
        QMessageBox.information(self,tr("drawing_export"),tr("drawing_export_saved"))

    def import_drawing(self):
        if not self.context.current_project:return
        path,_=QFileDialog.getOpenFileName(self,tr("drawing_import"),"",tr("drawing_images"))
        if path:
            try: self.select_drawing(self.context.drawings.import_drawing(self.context.current_project.project_id,path)); self.refresh()
            except (OSError,ValueError) as exc: QMessageBox.warning(self,tr("drawing_error"),tr("drawing_failure",detail=str(exc)))
    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,tr("drawing_model"),"",tr("drawing_weights"))
        if path:
            self.context.settings.drawing_model_path=path; self.context.settings.save(); self.refresh()
    def select_drawing(self,drawing):
        self.drawing=drawing; self.scene.clear(); self.pix=QGraphicsPixmapItem(QPixmap(drawing['managed_path'])); self.scene.addItem(self.pix); self.refresh_detections(); self.fit_view()
    def detect(self):
        if not self.drawing:return
        registry=DrawingModelRegistry(self.context.settings.drawing_model_path); detector=YOLODrawingDetector(registry.get())
        try: rows=self.context.drawings.save_detections(self.context.current_project.project_id,self.drawing['drawing_id'],registry.get().model_id,detector.detect(self.drawing['managed_path']))
        except Exception as exc: QMessageBox.warning(self,tr("drawing_error"),tr("drawing_failure",detail=str(exc))); return
        self.refresh_detections(rows)
    def refresh_detections(self,rows=None):
        if not self.drawing:return
        selected_id=self.selected()['detection_id'] if self.selected() else None
        chosen=self.map_equipment.currentData(); self.list.blockSignals(True)
        self.rows=rows if rows is not None else self.context.drawings.list_detections(self.context.current_project.project_id,self.drawing['drawing_id']); self.list.clear()
        equipment_names={item.equipment_id:item.name for item in self.context.equipment}
        for r in self.rows:
            mapped=equipment_names.get(r.get('equipment_id')) if r.get('review_status')=='confirmed' else None
            suffix=f" · {mapped}" if mapped else ""
            item=QListWidgetItem(f"{r['reviewed_class'] or r['class_name']}  {r['confidence']:.0%} · {tr('drawing_'+{'predicted':'prediction','confirmed':'confirmed','rejected':'rejected'}[r['review_status']])}{suffix}"); item.setData(Qt.UserRole,r['detection_id']); self.list.addItem(item)
        counts={};
        for r in self.rows: counts[r['class_name']]=counts.get(r['class_name'],0)+1
        self.summary.setText(" · ".join(f"{k}: {v}" for k,v in counts.items()) or tr("drawing_empty")); self.draw_boxes(); self.map_equipment.blockSignals(True); self.map_equipment.clear(); self.map_equipment.addItem(tr("drawing_choose_equipment"),None)
        for e in self.context.equipment: self.map_equipment.addItem(getattr(e,'name',None) or getattr(e,'equipment_name',None) or e.equipment_id,e.equipment_id)
        self.map_equipment.setCurrentIndex(max(0,self.map_equipment.findData(chosen))); self.map_equipment.blockSignals(False)
        self.list.setCurrentRow(next((i for i,r in enumerate(self.rows) if r["detection_id"]==selected_id),-1)); self.list.blockSignals(False)
        self.highlight(self.list.currentRow())
    def draw_boxes(self):
        for item in list(self.scene.items()):
            if item is not self.pix:self.scene.removeItem(item)
        names={item.equipment_id:item.name for item in self.context.equipment}; findings={item.equipment_id for item in (self.context.diagnosis_result.findings if self.context.diagnosis_result else [])}
        for r in self.rows:
            color=QColor(self.COLORS.get(r['class_name'].casefold(),"#7C3AED")); box=QGraphicsRectItem(r['bbox_x1'],r['bbox_y1'],r['bbox_x2']-r['bbox_x1'],r['bbox_y2']-r['bbox_y1']); box.setPen(QPen(color,3)); box.setData(0,r['detection_id']); self.scene.addItem(box)
            equipment=names.get(r.get('equipment_id')) if r.get('review_status')=='confirmed' else None
            status=" · "+tr("drawing_attention" if r.get('equipment_id') in findings else "drawing_no_finding" if self.context.diagnosis_result else "drawing_no_diagnosis") if equipment else ""
            text=QGraphicsTextItem(f"{r['reviewed_class'] or r['class_name']} {r['confidence']:.0%}" + (f" · {equipment}{status}" if equipment else "")); text.setData(0,r['detection_id']); text.setDefaultTextColor(color); text.setPos(r['bbox_x1'],r['bbox_y1']-20); self.scene.addItem(text)
    def highlight(self,index):
        if index>=0:
            target=self.rows[index]['detection_id']
            for item in self.scene.items():
                if isinstance(item,QGraphicsRectItem):
                    row=next((r for r in self.rows if r['detection_id']==item.data(0)),None)
                    item.setPen(QPen(QColor('#F59E0B') if item.data(0)==target else QColor(self.COLORS.get(row['class_name'].casefold(),'#7C3AED')),5 if item.data(0)==target else 3))
            row=self.rows[index]
            self.evidence_note.setText(tr('drawing_position',x=f"{(row['bbox_x1']+row['bbox_x2'])/2:.1f}",y=f"{(row['bbox_y1']+row['bbox_y2'])/2:.1f}")+'\n'+tr('drawing_evidence'))
    def selected(self): return self.rows[self.list.currentRow()] if 0<=self.list.currentRow()<len(self.rows) else None
    def review(self,status):
        row=self.selected()
        if not row:
            return
        self.context.drawings.review_detection(
            self.context.current_project.project_id, row['detection_id'], status,
            self.class_choice.currentText() if status == 'confirmed' else None,
        )
        self.refresh_detections()
    def map_selected(self):
        row=self.selected()
        if not row or not self.map_equipment.currentData():return
        try:self.context.drawings.map_equipment(self.context.current_project.project_id,row['detection_id'],self.map_equipment.currentData())
        except ValueError as exc: QMessageBox.warning(self,tr("drawing_association"),tr("drawing_failure",detail=str(exc))); return
        self.refresh_detections()
    def open_selected_equipment(self):
        row=self.selected()
        if not row or row.get('review_status')!='confirmed' or not row.get('equipment_id'):return
        equipment=next((item for item in self.context.equipment if item.equipment_id==row['equipment_id']),None)
        if equipment:self.navigation_requested.emit("equipment",{"equipment_id":equipment.name})
    def apply_global_context(self):
        return None
    def fit_view(self):
        if self.scene.items():self.viewer.fitInView(self.scene.itemsBoundingRect(),Qt.KeepAspectRatio)

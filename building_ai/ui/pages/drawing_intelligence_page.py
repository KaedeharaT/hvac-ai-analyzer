from __future__ import annotations
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor, QPen, QPixmap
from PyQt5.QtWidgets import QComboBox, QFileDialog, QGraphicsPixmapItem, QGraphicsRectItem, QGraphicsScene, QGraphicsTextItem, QGraphicsView, QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSplitter, QVBoxLayout, QWidget
from building_ai.i18n import LanguageManager
from building_ai.vision import DrawingModelRegistry, YOLODrawingDetector

class DrawingIntelligencePage(QWidget):
    navigation_requested = pyqtSignal(str, object)
    COLORS={"aircon":"#2563EB","window":"#059669","baseline":"#64748B","baseline_mark":"#64748B"}
    def __init__(self, context):
        super().__init__(); self.context=context; self.drawing=None; self.rows=[]
        root=QVBoxLayout(self); root.setContentsMargins(24,24,24,24)
        self.title=QLabel(); self.title.setObjectName("PageTitle"); root.addWidget(self.title); self.subtitle=QLabel(); self.subtitle.setObjectName("Muted"); self.subtitle.setWordWrap(True); root.addWidget(self.subtitle)
        bar=QHBoxLayout(); self.import_button=QPushButton(); self.model_button=QPushButton(); self.detect_button=QPushButton(); self.zoom_out=QPushButton("−"); self.fit=QPushButton("Fit"); self.zoom_in=QPushButton("+"); self.model_status=QLabel(); self.model_status.setObjectName("StatusBadge")
        for x in (self.import_button,self.model_button,self.detect_button,self.zoom_out,self.fit,self.zoom_in,self.model_status): bar.addWidget(x)
        bar.addStretch(1); root.addLayout(bar); split=QSplitter(); self.scene=QGraphicsScene(self); self.viewer=QGraphicsView(self.scene); self.viewer.setDragMode(QGraphicsView.ScrollHandDrag); split.addWidget(self.viewer)
        side=QWidget(); sidebox=QVBoxLayout(side); self.summary=QLabel(); self.summary.setWordWrap(True); self.list=QListWidget(); self.class_choice=QComboBox(); self.class_choice.addItems(["aircon","baseline_mark","window"]); self.confirm=QPushButton(); self.reject=QPushButton(); self.map_equipment=QComboBox(); self.map_button=QPushButton(); self.open_equipment=QPushButton(); self.open_equipment.setObjectName("PrimaryButton"); self.open_equipment.clicked.connect(self.open_selected_equipment)
        self.review_label=QLabel(); self.mapping_label=QLabel(); self.evidence_note=QLabel(); self.evidence_note.setObjectName("Muted"); self.evidence_note.setWordWrap(True)
        for x in (self.summary,self.list,self.review_label,self.class_choice,self.confirm,self.reject,self.mapping_label,self.map_equipment,self.map_button,self.open_equipment,self.evidence_note): sidebox.addWidget(x)
        split.addWidget(side); split.setSizes([850,330]); root.addWidget(split,1)
        self.import_button.clicked.connect(self.import_drawing); self.model_button.clicked.connect(self.choose_model); self.detect_button.clicked.connect(self.detect); self.zoom_in.clicked.connect(lambda:self.viewer.scale(1.2,1.2)); self.zoom_out.clicked.connect(lambda:self.viewer.scale(.8,.8)); self.fit.clicked.connect(self.fit_view); self.list.currentRowChanged.connect(self.highlight); self.confirm.clicked.connect(lambda:self.review("confirmed")); self.reject.clicked.connect(lambda:self.review("rejected")); self.map_button.clicked.connect(self.map_selected); LanguageManager.instance().language_changed.connect(self.retranslate_ui); self.retranslate_ui()
    def retranslate_ui(self):
        zh=LanguageManager.instance().language=="zh_CN"; self.title.setText("图纸与空间信息" if zh else "Drawing & Layout Intelligence"); self.subtitle.setText("CV 检测、人工复核与设备运行状态保持为独立证据；只有已确认关联才会显示设备信息。" if zh else "CV detections, human review, and BEMS status remain separate evidence. Equipment context appears only for confirmed associations."); self.import_button.setText("导入图纸" if zh else "Import Drawing"); self.model_button.setText("选择模型" if zh else "Choose Model"); self.detect_button.setText("开始识别" if zh else "Run Detection"); self.confirm.setText("确认" if zh else "Confirm"); self.reject.setText("拒绝" if zh else "Reject"); self.map_button.setText("保存关联" if zh else "Save Association"); self.review_label.setText("人工复核类别" if zh else "Human review class"); self.mapping_label.setText("已确认设备关联" if zh else "Confirmed equipment association"); self.open_equipment.setText("打开设备详情" if zh else "Open Equipment Detail"); self.evidence_note.setText("设备状态来自 BEMS 确定性分析，不由图纸检测推断。" if zh else "Equipment status comes from deterministic BEMS analysis, never from the drawing detector.")
    def refresh(self):
        if not self.context.current_project: self.model_status.setText("请选择项目 / Select a project"); return
        zh=LanguageManager.instance().language=="zh_CN"; registry=DrawingModelRegistry(self.context.settings.drawing_model_path); self.model_status.setText(("Legacy YOLOv8 图纸检测器 · 已就绪" if zh else "Legacy YOLOv8 Drawing Detector · Ready") if registry.configured() else ("图纸识别模型未配置" if zh else "Model not configured"))
        drawings=self.context.drawings.list_drawings(self.context.current_project.project_id)
        if drawings and (self.drawing is None or self.drawing['drawing_id'] not in {d['drawing_id'] for d in drawings}): self.select_drawing(drawings[0])
    def import_drawing(self):
        if not self.context.current_project:return
        path,_=QFileDialog.getOpenFileName(self,"Import drawing","","Images (*.png *.jpg *.jpeg)")
        if path:self.select_drawing(self.context.drawings.import_drawing(self.context.current_project.project_id,path))
    def choose_model(self):
        path,_=QFileDialog.getOpenFileName(self,"Select drawing model","","YOLO weights (*.pt)")
        if path:
            self.context.settings.drawing_model_path=path; self.context.settings.save(); self.refresh()
    def select_drawing(self,drawing):
        self.drawing=drawing; self.scene.clear(); self.pix=QGraphicsPixmapItem(QPixmap(drawing['managed_path'])); self.scene.addItem(self.pix); self.refresh_detections(); self.fit_view()
    def detect(self):
        if not self.drawing:return
        registry=DrawingModelRegistry(self.context.settings.drawing_model_path); detector=YOLODrawingDetector(registry.get())
        try: rows=self.context.drawings.save_detections(self.context.current_project.project_id,self.drawing['drawing_id'],registry.get().model_id,detector.detect(self.drawing['managed_path']))
        except Exception as exc: QMessageBox.warning(self,"Drawing model",str(exc)); return
        self.refresh_detections(rows)
    def refresh_detections(self,rows=None):
        if not self.drawing:return
        self.rows=rows if rows is not None else self.context.drawings.list_detections(self.context.current_project.project_id,self.drawing['drawing_id']); self.list.clear()
        equipment_names={item.equipment_id:item.name for item in self.context.equipment}
        for r in self.rows:
            mapped=equipment_names.get(r.get('equipment_id')) if r.get('review_status')=='confirmed' else None
            suffix=f" · {mapped}" if mapped else ""
            item=QListWidgetItem(f"{r['reviewed_class'] or r['class_name']}  {r['confidence']:.0%} · {r['review_status']}{suffix}"); item.setData(Qt.UserRole,r['detection_id']); self.list.addItem(item)
        counts={};
        for r in self.rows: counts[r['class_name']]=counts.get(r['class_name'],0)+1
        self.summary.setText(" · ".join(f"{k}: {v}" for k,v in counts.items()) or "No detections"); self.draw_boxes(); self.map_equipment.clear()
        for e in self.context.equipment: self.map_equipment.addItem(getattr(e,'name',None) or getattr(e,'equipment_name',None) or e.equipment_id,e.equipment_id)
    def draw_boxes(self):
        for item in list(self.scene.items()):
            if item is not self.pix:self.scene.removeItem(item)
        names={item.equipment_id:item.name for item in self.context.equipment}; findings={item.equipment_id for item in (self.context.diagnosis_result.findings if self.context.diagnosis_result else [])}
        for r in self.rows:
            color=QColor(self.COLORS.get(r['class_name'].casefold(),"#7C3AED")); box=QGraphicsRectItem(r['bbox_x1'],r['bbox_y1'],r['bbox_x2']-r['bbox_x1'],r['bbox_y2']-r['bbox_y1']); box.setPen(QPen(color,3)); box.setData(0,r['detection_id']); self.scene.addItem(box)
            equipment=names.get(r.get('equipment_id')) if r.get('review_status')=='confirmed' else None
            zh=LanguageManager.instance().language=="zh_CN"; status=((" · 需关注" if r.get('equipment_id') in findings else " · 当前无诊断结果") if zh else (" · Needs attention" if r.get('equipment_id') in findings else " · No active finding")) if equipment else ""
            text=QGraphicsTextItem(f"{r['reviewed_class'] or r['class_name']} {r['confidence']:.0%}" + (f" · {equipment}{status}" if equipment else "")); text.setDefaultTextColor(color); text.setPos(r['bbox_x1'],r['bbox_y1']-20); self.scene.addItem(text)
    def highlight(self,index):
        if index>=0:
            target=self.rows[index]['detection_id']
            for item in self.scene.items():
                if item.data(0)==target:item.setPen(QPen(QColor('#F59E0B'),5)); self.viewer.ensureVisible(item)
    def selected(self): return self.rows[self.list.currentRow()] if self.list.currentRow()>=0 else None
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
        if not row or self.map_equipment.currentIndex()<0:return
        try:self.context.drawings.map_equipment(self.context.current_project.project_id,row['detection_id'],self.map_equipment.currentData())
        except ValueError as exc: QMessageBox.warning(self,"Drawing association",str(exc)); return
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

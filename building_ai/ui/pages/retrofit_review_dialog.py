"""Existing-building energy advice entry; source data and computations stay intact."""
import json
from PyQt5.QtCore import Qt, pyqtSlot
from PyQt5.QtWidgets import QDialog, QVBoxLayout, QHBoxLayout, QTextEdit, QPushButton, QFileDialog, QMessageBox
from building_ai.i18n import LanguageManager, tr
from building_ai.services.retrofit_review_renderer import render_review


class RetrofitReviewDialog(QDialog):
    def __init__(self, context, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WA_DeleteOnClose)
        self.context=context
        self.project_id=context.current_project.project_id
        self.setWindowModality(Qt.WindowModal)
        self.resize(900,680)
        layout=QVBoxLayout(self)
        self.body=QTextEdit(); self.body.setReadOnly(True); layout.addWidget(self.body)
        buttons=QHBoxLayout(); self.export_button=QPushButton(); self.close_button=QPushButton()
        buttons.addWidget(self.export_button); buttons.addStretch(); buttons.addWidget(self.close_button); layout.addLayout(buttons)
        self.export_button.clicked.connect(self.export); self.close_button.clicked.connect(self.close)
        self.report=context.agent.get_retrofit_review(self.project_id,context.selected_equipment_id)
        LanguageManager.instance().language_changed.connect(self.retranslate_ui)
        self.retranslate_ui()

    @pyqtSlot()
    def retranslate_ui(self):
        self.setWindowTitle(tr('retrofit_title')); self.export_button.setText(tr('retrofit_export')); self.close_button.setText(tr('retrofit_close'))
        self.body.setPlainText(render_review(self.report,LanguageManager.instance().language))

    def export(self):
        path,_=QFileDialog.getSaveFileName(self,tr('retrofit_export'),'buildingai-retrofit-review.json','JSON (*.json)')
        if not path: return
        try:
            with open(path,'w',encoding='utf-8') as handle: json.dump(self.report,handle,ensure_ascii=False,indent=2,allow_nan=False)
        except (OSError,ValueError) as exc: QMessageBox.warning(self,tr('report_failed'),str(exc))

"""Optional Qt5 initialization; no browser process or model is opened here."""
def prepare_model_viewer() -> bool:
    from PyQt5.QtCore import Qt, QCoreApplication
    QCoreApplication.setAttribute(Qt.AA_ShareOpenGLContexts)
    try:
        # Older Qt5 WebEngine builds require import before QApplication, even
        # with shared contexts. Missing extras must never block the BEMS app.
        from PyQt5 import QtWebEngineWidgets  # noqa: F401
    except (ImportError, RuntimeError):
        return False
    return True

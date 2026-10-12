# Model Viewer: third-party inventory

The optional preview uses one rendering stack. This inventory records the
versions reviewed for integration; it does not change upstream license terms.

| Component | Use / distribution | License and notice |
| --- | --- | --- |
| Three.js 0.186.1, including OrbitControls | Compiled into the offline `viewer.bundle.js` | MIT; [bundled notice](../building_ai/ui/model_viewer_assets/THREE-LICENSE.txt) |
| Mapbox earcut 3.0.2 | Copied inside Three.js; used to triangulate source surface polygons | ISC; [bundled notice](../building_ai/ui/model_viewer_assets/EARCUT-LICENSE.txt) |
| esbuild 0.28.2 | Pinned development/build tool; its executable is not distributed in this repository | MIT; [upstream license](https://github.com/evanw/esbuild/blob/v0.28.2/LICENSE.md) |
| PyQtWebEngine >= 5.15.4, < 6 | Optional Python package installed by the user; no binary bundled here | GPL v3 or Riverbank commercial license; [publisher terms](https://www.riverbankcomputing.com/software/pyqtwebengine/intro) |
| Qt WebEngine / WebChannel and Chromium dependencies | Supplied by the installed Qt distribution, including its `qrc` WebChannel resource | Retain the installed distribution's notices; [Qt licensing inventory](https://doc.qt.io/qt-5/qtwebengine-licensing.html) |

The project's own source license does not relicense these dependencies. This
change publishes source and the licensed Three.js bundle, not a packaged Qt
runtime or a new desktop Release. A packaged distribution must retain the
applicable Qt/PyQt/Chromium terms and notices.

The Earcut version was identified in Three.js's
`src/extras/lib/earcut.js`; its ISC text comes from the
[versioned upstream license](https://github.com/mapbox/earcut/blob/v3.0.2/LICENSE).
The Three.js notice and its generated bundle's MIT banner are retained. The
versioned build dependencies and npm integrity values are in
`tools/model_viewer/package-lock.json`.

Runtime HTML, JavaScript and notices are shipped as package data. The browser
permits only local viewer assets and Qt's WebChannel script. Network connections,
remote navigation, external images and workers are blocked. Model names and
review notes are displayed as plain text; geometry is passed as JSON, without
HTML interpolation or execution of OSM content.

Rebuild with `npm ci --prefix tools/model_viewer --ignore-scripts` followed by
`npm run --prefix tools/model_viewer build`. Node/npm are build tools only.
No external JavaScript, model, texture, font or CDN is needed to preview a model.
